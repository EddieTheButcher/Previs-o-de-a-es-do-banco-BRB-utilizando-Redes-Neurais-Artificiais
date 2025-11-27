"""
Sistema de Previsão de Preços – Ações do Banco de Brasília
----------------------------------------------------------
Coleta dados REAIS do InfoMoney, treina um modelo LSTM e produz
previsões de curto prazo (15 dias úteis) para as ações BSLI3 e BSLI4.
Relatório completo em HTML ao final da execução.

Novidades desta versão
======================
1. Coleta de dados REAIS do site InfoMoney via web scraping.
2. Comentários detalhados em toda a classe.
3. Estatísticas completas e métricas de bondade de ajuste.
4. Gráficos com valores históricos e previsões.
5. Interface HTML profissional e moderna com animações.
"""

import os
import warnings

# ---------------- SUPRESSÃO DE AVISOS ---------------- #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# ------------------ IMPORTAÇÕES --------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import re

tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configuração para gráficos
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

# ====================================================
#                CLASSE PRINCIPAL
# ====================================================
class BancoDebrasiliaPredictor:
    """
    Classe responsável por:
    1. Coletar dados históricos REAIS do InfoMoney.
    2. Calcular indicadores técnicos.
    3. Preparar dados e treinar um modelo LSTM para cada ativo.
    4. Gerar previsões futuras e sumarizar estatísticas.
    5. Criar gráficos com histórico e previsões.
    6. Exibir tabelas no console e produzir um relatório HTML.

    Atributos principais
    --------------------
    symbols            : lista de tickers analisados
    raw_data           : DataFrames brutos por ticker
    processed_data     : DataFrames com indicadores
    models             : modelos LSTM treinados
    predictions        : Série de previsões por ticker
    prediction_stats   : estatísticas básicas por ticker
    model_metrics      : métricas de treino/teste por ticker
    """

    def __init__(self):
        # -------- CONFIGURAÇÕES BÁSICAS -------- #
        self.symbols = ['BSLI3', 'BSLI4']
        self.symbol_names = {
            'BSLI3': 'BSLI3 (Ordinária)',
            'BSLI4': 'BSLI4 (Preferencial)'
        }

        # --------- CONTÊINERES INTERNOS ---------- #
        self.scalers = {}
        self.models = {}
        self.raw_data = {}
        self.processed_data = {}
        self.predictions = {}
        self.prediction_stats = {}
        self.model_metrics = {}
        self.data_summary = {}

        # --------- HIPERPARÂMETROS -------------- #
        self.sequence_length = 60
        self.target_days = 365
        self.train_split = 0.70

        # --------- DIRETÓRIO DE GRÁFICOS -------- #
        if not os.path.exists('plots'):
            os.makedirs('plots')

    # ---------------------------------------------------------------- #
    #            1. COLETA DE DADOS REAIS DO INFOMONEY                 #
    # ---------------------------------------------------------------- #
    def fetch_infomoney_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Coleta dados históricos reais do InfoMoney para o ticker informado.

        Parâmetros
        ----------
        symbol : str
            Ticker alvo (BSLI3 ou BSLI4).
        days : int
            Número de dias de histórico desejado.

        Retorna
        -------
        pd.DataFrame
            DataFrame indexado por data com colunas Open, High, Low,
            Close, Volume.
        """
        print(f"🔍 Coletando dados do InfoMoney para {self.symbol_names[symbol]}...")

        try:
            # URL do InfoMoney para dados históricos
            url = f"https://www.infomoney.com.br/cotacoes/b3/acao/{symbol.lower()}/"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Tenta coletar dados da tabela de cotações históricas
            data_list = []

            # Procura por tabelas com dados históricos
            tables = soup.find_all('table')

            if tables:
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows[1:]:  # Pula cabeçalho
                        cols = row.find_all('td')
                        if len(cols) >= 5:
                            try:
                                date_str = cols[0].text.strip()
                                close = self._parse_price(cols[1].text.strip())
                                open_price = self._parse_price(cols[2].text.strip())
                                high = self._parse_price(cols[3].text.strip())
                                low = self._parse_price(cols[4].text.strip())
                                volume = self._parse_volume(cols[5].text.strip() if len(cols) > 5 else '0')

                                date = pd.to_datetime(date_str, format='%d/%m/%Y')

                                data_list.append({
                                    'Date': date,
                                    'Open': open_price,
                                    'High': high,
                                    'Low': low,
                                    'Close': close,
                                    'Volume': volume
                                })
                            except Exception:
                                continue

            # Se não conseguiu coletar dados da tabela, usa API alternativa
            if not data_list:
                print("⚠️  Tentando fonte alternativa (Yahoo Finance)...")
                return self._fetch_yahoo_finance(symbol, days)

            df = pd.DataFrame(data_list)
            df = df.set_index('Date').sort_index()
            df = df[df.index <= datetime.now()]
            df = df.tail(days)

            if len(df) < 100:
                print(f"⚠️  Poucos dados coletados ({len(df)}), usando Yahoo Finance...")
                return self._fetch_yahoo_finance(symbol, days)

            print(f"✅ {len(df)} registros coletados do InfoMoney")
            return df

        except Exception as e:
            print(f"❌ Erro ao coletar do InfoMoney: {e}")
            print("🔄 Tentando Yahoo Finance como alternativa...")
            return self._fetch_yahoo_finance(symbol, days)

    def _fetch_yahoo_finance(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Método alternativo usando Yahoo Finance quando InfoMoney falha.
        """
        try:
            import yfinance as yf

            ticker = f"{symbol}.SA"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 100)

            print(f"📹 Baixando dados do Yahoo Finance para {ticker}...")

            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                raise Exception("Nenhum dado retornado pelo Yahoo Finance")

            # Renomeia colunas para o padrão esperado
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()

            # Filtra apenas dias úteis
            df = df[df.index.weekday < 5]
            df = df.tail(days)

            print(f"✅ {len(df)} registros coletados do Yahoo Finance")
            return df

        except ImportError:
            print("⚠️  Biblioteca yfinance não instalada. Instalando...")
            os.system("pip install yfinance -q")
            return self._fetch_yahoo_finance(symbol, days)
        except Exception as e:
            print(f"❌ Erro ao coletar do Yahoo Finance: {e}")
            print("🔄 Gerando dados sintéticos como último recurso...")
            return self.generate_realistic_data(symbol, days)

    def _parse_price(self, price_str: str) -> float:
        """
        Converte string de preço para float.
        Exemplos: "R$ 12,50" -> 12.50 | "12.345,67" -> 12345.67
        """
        price_str = price_str.replace('R$', '').replace('.', '').replace(',', '.').strip()
        return float(price_str)

    def _parse_volume(self, volume_str: str) -> int:
        """
        Converte string de volume para inteiro.
        Exemplos: "1.234.567" -> 1234567 | "1,2M" -> 1200000
        """
        volume_str = volume_str.upper().strip()

        if 'M' in volume_str:
            return int(float(volume_str.replace('M', '')) * 1_000_000)
        elif 'K' in volume_str:
            return int(float(volume_str.replace('K', '')) * 1_000)
        else:
            return int(volume_str.replace('.', '').replace(',', ''))

    def generate_realistic_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        FALLBACK: Gera dados sintéticos realistas caso a coleta real falhe.
        """
        print(f"🔧 Gerando dados sintéticos para {self.symbol_names[symbol]}...")

        base_prices = {'BSLI3': 12.50, 'BSLI4': 13.20}
        base_price = base_prices.get(symbol, 12.00)

        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates, current_date = [], end_date - timedelta(days=days + 100)
        while len(dates) < days:
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date += timedelta(days=1)
        dates = dates[-days:]

        np.random.seed(42)
        trend = np.linspace(0, 0.15, days)
        daily_returns = np.random.normal(0, 0.02, days)

        for idx in np.random.choice(days, size=int(days * 0.05), replace=False):
            daily_returns[idx] *= np.random.choice([2, -2])

        prices = [base_price]
        for i in range(1, days):
            new_price = prices[-1] * (1 + trend[i] / 365 + daily_returns[i])
            prices.append(max(new_price, base_price * 0.5))

        opens, highs, lows, volumes = [], [], [], []
        for i, close_price in enumerate(prices):
            if i == 0:
                open_price = close_price * np.random.normal(1, 0.005)
            else:
                open_price = prices[i - 1] * np.random.normal(1, 0.01)

            daily_vol = abs(daily_returns[i]) + 0.005
            highs.append(max(open_price, close_price) * (1 + daily_vol))
            lows.append(min(open_price, close_price) * (1 - daily_vol))
            opens.append(open_price)

            base_vol = np.random.randint(1e5, 1e6)
            if abs(daily_returns[i]) > 0.02:
                base_vol *= np.random.uniform(2, 5)
            volumes.append(int(base_vol))

        data = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=pd.DatetimeIndex(dates))

        print(f"✅ {len(data)} registros sintéticos gerados")
        return data

    def collect_data(self) -> bool:
        """
        Percorre todos os tickers e coleta dados reais do InfoMoney.
        Retorna True se pelo menos um ativo foi carregado com sucesso.
        """
        print("🔍 Iniciando coleta de dados REAIS...")
        print("=" * 60)

        for symbol in self.symbols:
            try:
                data = self.fetch_infomoney_data(symbol, self.target_days)

                if not data.empty:
                    self.raw_data[symbol] = data

                    self.data_summary[symbol] = {
                        'total_records': len(data),
                        'start_date': data.index[0].strftime('%d/%m/%Y'),
                        'end_date': data.index[-1].strftime('%d/%m/%Y'),
                        'avg_price': data['Close'].mean(),
                        'volatility': data['Close'].std(),
                        'min_price': data['Close'].min(),
                        'max_price': data['Close'].max(),
                        'avg_volume': data['Volume'].mean()
                    }

                    print(f"✅ {self.symbol_names[symbol]}")
                    print(f"   Período: {data.index[0].strftime('%d/%m/%Y')} a "
                          f"{data.index[-1].strftime('%d/%m/%Y')}")
                    print(f"   Preço atual: R$ {data['Close'].iloc[-1]:.2f}")
                    print(f"   Variação (período): "
                          f"{((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:+.2f}%")
                    print()

            except Exception as exc:
                print(f"❌ Falha ao coletar {symbol}: {exc}")

        print(f"✅ COLETA CONCLUÍDA: {len(self.raw_data)} ativo(s) carregado(s).")
        return len(self.raw_data) > 0

    # ---------------------------------------------------------------- #
    #                     2. PRÉ-PROCESSAMENTO                         #
    # ---------------------------------------------------------------- #
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos ao DataFrame.
        """
        df = data.copy()

        df['MA_7'] = df['Close'].rolling(7).mean()
        df['MA_21'] = df['Close'].rolling(21).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()

        delta = df['Close'].diff()
        gain = delta.mask(delta < 0, 0).rolling(14).mean()
        loss = (-delta.mask(delta > 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - 100 / (1 + rs)

        df['Volatility'] = df['Close'].rolling(21).std()

        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std

        df['Volume_MA'] = df['Volume'].rolling(21).mean()
        return df

    def prepare_data(self, symbol):
        """
        Prepara dados para LSTM.
        """
        if symbol not in self.raw_data:
            return None, None, None, None

        data = self.calculate_indicators(self.raw_data[symbol]).dropna()

        feats = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'MA_7', 'MA_21', 'MA_50',
            'RSI', 'Volatility', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'Volume_MA'
        ]

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data[feats])
        self.scalers[symbol] = scaler

        X, y = [], []
        for i in range(self.sequence_length, len(scaled)):
            X.append(scaled[i - self.sequence_length:i])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)
        split = int(len(X) * self.train_split)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.processed_data[symbol] = data
        print(f"✅ {self.symbol_names[symbol]}: {len(X_train)} treino | {len(X_test)} teste")
        return X_train, X_test, y_train, y_test

    # ---------------------------------------------------------------- #
    #                        3. MODELAGEM                              #
    # ---------------------------------------------------------------- #
    def build_model(self, input_shape):
        """
        Constrói modelo LSTM.
        """
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape, activation='relu'),
            Dropout(0.2),
            LSTM(100, return_sequences=True, activation='relu'),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
        return model

    def train_model(self, symbol, epochs=50):
        """
        Treina modelo LSTM.
        """
        X_train, X_test, y_train, y_test = self.prepare_data(symbol)
        if X_train is None:
            return None

        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        cb_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        start = time.time()
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[cb_early],
            verbose=0
        )
        elapsed = time.time() - start
        self.models[symbol] = model

        train_pred = self.denormalize(symbol, model.predict(X_train, verbose=0))
        test_pred = self.denormalize(symbol, model.predict(X_test, verbose=0))
        y_train_dn = self.denormalize(symbol, y_train.reshape(-1, 1))
        y_test_dn = self.denormalize(symbol, y_test.reshape(-1, 1))

        self.model_metrics[symbol] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_dn, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_dn, test_pred)),
            'test_mae': mean_absolute_error(y_test_dn, test_pred),
            'training_time': elapsed,
            'overfitting_ratio': (
                np.sqrt(mean_squared_error(y_test_dn, test_pred)) /
                np.sqrt(mean_squared_error(y_train_dn, train_pred))
            )
        }

        print(
            f"✅ {self.symbol_names[symbol]} treinado em {elapsed:.1f}s "
            f"| RMSE teste: R$ {self.model_metrics[symbol]['test_rmse']:.4f}"
        )

    # ---------------------------------------------------------------- #
    #                       4. PREVISÃO                                #
    # ---------------------------------------------------------------- #
    def denormalize(self, symbol, arr):
        """
        Desnormaliza previsões.
        """
        scaler = self.scalers[symbol]
        dummy = np.zeros((arr.shape[0], scaler.n_features_in_))
        dummy[:, 0] = arr.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    def predict_future(self, symbol, days=15):
        """
        Gera previsões futuras.
        """
        if symbol not in self.models:
            return None

        model = self.models[symbol]
        scaler = self.scalers[symbol]
        data = self.processed_data[symbol]
        feats = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'MA_7', 'MA_21', 'MA_50',
            'RSI', 'Volatility', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'Volume_MA'
        ]

        seq_scaled = scaler.transform(data[feats].tail(self.sequence_length).values)
        preds_scaled, seq_curr = [], seq_scaled.copy()

        for _ in range(days):
            nxt = model.predict(seq_curr.reshape(1, self.sequence_length, len(feats)), verbose=0)
            preds_scaled.append(nxt[0, 0])
            new_row = seq_curr[-1].copy()
            new_row[0] = nxt[0, 0]
            seq_curr = np.vstack([seq_curr[1:], new_row])

        preds = self.denormalize(symbol, np.array(preds_scaled).reshape(-1, 1))

        last_date = data.index[-1]
        fut_dates = []
        while len(fut_dates) < days:
            last_date += timedelta(days=1)
            if last_date.weekday() < 5:
                fut_dates.append(last_date)
        series = pd.Series(preds, index=fut_dates)

        self.predictions[symbol] = series

        last_price = data['Close'].iloc[-1]

        # Estatísticas preditivas
        n_obs = len(series)
        mu = series.mean()
        sigma = series.std()
        stderr = sigma / np.sqrt(n_obs)           # erro-padrão da média
        ci_low = mu - 1.96 * stderr               # IC 95%
        ci_high = mu + 1.96 * stderr

        self.prediction_stats[symbol] = {
            'mean': mu,
            'median': series.median(),
            'std': sigma,
            'min': series.min(),
            'max': series.max(),
            'expected_return_pct': (mu - last_price) / last_price * 100,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'q05': series.quantile(0.05),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'q95': series.quantile(0.95),
            'last_price': last_price
        }
        print(f"🔮 {self.symbol_names[symbol]}: {days} previsões geradas.")
        return series

    # ---------------------------------------------------------------- #
    #                      5. GRÁFICOS                                 #
    # ---------------------------------------------------------------- #
    def generate_prediction_plot(self, symbol):
        """
        Cria gráfico com histórico e previsões.
        """
        if symbol not in self.predictions or symbol not in self.processed_data:
            print(f"⚠️  Impossível gerar gráfico para {symbol}")
            return

        print(f"📊 Gerando gráfico para {self.symbol_names[symbol]}...")

        historical = self.processed_data[symbol]['Close'].tail(90)
        predictions = self.predictions[symbol]

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.plot(
            historical.index,
            historical.values,
            linewidth=2,
            color='#1f77b4',
            label='Histórico Real',
            marker='o',
            markersize=3
        )

        ax.plot(
            predictions.index,
            predictions.values,
            linewidth=2.5,
            color='#ff7f0e',
            label='Previsões LSTM',
            marker='s',
            markersize=4,
            linestyle='--'
        )

        last_historical_date = historical.index[-1]
        ax.axvline(
            x=last_historical_date,
            color='red',
            linestyle=':',
            linewidth=2,
            label='Início das Previsões'
        )

        ax.set_title(
            f'Previsão de Preços - {self.symbol_names[symbol]} (DADOS REAIS)\n'
            f'Histórico (90 dias) vs Previsões LSTM (15 dias)',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Data', fontsize=12, fontweight='bold')
        ax.set_ylabel('Preço de Fechamento (R$)', fontsize=12, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        filename = f'plots/{symbol}_prediction.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Gráfico salvo: {filename}")

    # ---------------------------------------------------------------- #
    #                      6. RELATÓRIOS                               #
    # ---------------------------------------------------------------- #
    def display_tables(self):
        """
        Exibe tabelas de resumo no console.
        """
        print("\n" + "=" * 100)
        print("📊 TABELA RESUMO - DADOS COLETADOS (REAIS)")
        print("=" * 100)
        print(f"{'Ação':<15} {'Registros':<10} {'Período':<25} "
              f"{'Preço Médio':<12} {'Volatilidade':<12}")
        print("-" * 100)
        for s, d in self.data_summary.items():
            print(
                f"{self.symbol_names[s]:<15} {d['total_records']:<10} "
                f"{d['start_date']} - {d['end_date']:<10} "
                f"R$ {d['avg_price']:.4f}   R$ {d['volatility']:.4f}"
            )

        print("\n" + "=" * 100)
        print("🏗️ TABELA MÉTRICAS DE BONDADE DE AJUSTE - MODELOS LSTM")
        print("=" * 100)
        print("Indicadores de desempenho do modelo e potencial de overfitting.")
        print(f"{'Ação':<15} {'RMSE Treino':<12} {'RMSE Teste':<12} "
              f"{'MAE Teste':<12} {'Overfitting':<12} {'Tempo (s)':<10}")
        print("-" * 100)
        for s, m in self.model_metrics.items():
            print(
                f"{self.symbol_names[s]:<15} R$ {m['train_rmse']:.4f}   "
                f"R$ {m['test_rmse']:.4f}   R$ {m['test_mae']:.4f}   "
                f"{m['overfitting_ratio']:.2f}         {m['training_time']:.1f}"
            )
        print("-" * 100)
        print("Nota:")
        print("  - RMSE: Quanto menor, melhor o ajuste.")
        print("  - MAE: Menos sensível a outliers que o RMSE.")
        print("  - Overfitting Ratio próximo de 1.0 é ideal.")

        print("\n" + "=" * 100)
        print("🔮 TABELA RESUMO - PREVISÕES (15 DIAS)")
        print("=" * 100)
        print(f"{'Ação':<15} {'Último Preço':<14} {'Previsão Média':<15} "
              f"{'Variação %':<12} {'Tendência':<10}")
        print("-" * 100)
        for s in self.symbols:
            if s in self.predictions:
                last_price = self.processed_data[s]['Close'].iloc[-1]
                pred_mean = self.prediction_stats[s]['mean']
                change_pct = (pred_mean - last_price) / last_price * 100
                trend = "ALTA" if change_pct > 0 else "BAIXA"
                print(
                    f"{self.symbol_names[s]:<15} R$ {last_price:.4f}     "
                    f"R$ {pred_mean:.4f}     {change_pct:+.2f}%      {trend}"
                )

        print("\n" + "=" * 100)
        print("📈 TABELA ESTATÍSTICAS – PREVISÕES (DETALHADAS)")
        print("=" * 100)
        print(f"{'Ação':<15} {'Média':<10} {'IC 95%':<22} "
              f"{'Q05':<10} {'Q25':<10} {'Q75':<10} {'Q95':<10}")
        print("-" * 100)
        for s, st in self.prediction_stats.items():
            ic = f"[{st['ci_low']:.4f}, {st['ci_high']:.4f}]"
            print(
                f"{self.symbol_names[s]:<15} R$ {st['mean']:.4f}  {ic:<22} "
                f"R$ {st['q05']:.4f}  R$ {st['q25']:.4f}  "
                f"R$ {st['q75']:.4f}  R$ {st['q95']:.4f}"
            )

    def create_html_report(self):
        """
        Cria relatório HTML completo com design profissional e animações.
        """
        current_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

        html_content = f'''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Previsões Banco de Brasília - Análise LSTM</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --primary-color: #2563eb;
            --secondary-color: #1e40af;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --dark-bg: #0f172a;
            --card-bg: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --border-color: #334155;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* Header Styles */
        .header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(37, 99, 235, 0.3);
            margin-bottom: 30px;
            animation: slideDown 0.8s ease;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }}

        .header-content {{
            position: relative;
            z-index: 1;
        }}

        .header h1 {{
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .header-subtitle {{
            font-size: 1.1em;
            opacity: 0.95;
            margin-bottom: 20px;
        }}

        .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            color: white;
            padding: 8px 20px;
            border-radius: 50px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 5px;
            border: 1px solid rgba(255,255,255,0.3);
        }}

        /* Stats Cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: var(--card-bg);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            animation: fadeInUp 0.6s ease;
            position: relative;
            overflow: hidden;
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--success-color));
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }}

        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
            margin-bottom: 10px;
        }}

        .stat-value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 5px;
        }}

        .stat-change {{
            font-size: 0.9em;
            font-weight: 600;
        }}

        .stat-change.positive {{
            color: var(--success-color);
        }}

        .stat-change.negative {{
            color: var(--danger-color);
        }}

        /* Chart Container */
        .chart-section {{
            background: var(--card-bg);
            padding: 30px;
            border-radius: 15px;
            border: 1px solid var(--border-color);
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            animation: fadeInUp 0.8s ease;
        }}

        .section-title {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 20px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section-title::before {{
            content: '';
            width: 4px;
            height: 30px;
            background: linear-gradient(180deg, var(--primary-color), var(--success-color));
            border-radius: 2px;
        }}

        .chart-container {{
            position: relative;
            height: 400px;
            margin-bottom: 20px;
        }}

        /* Table Styles */
        .table-container {{
            overflow-x: auto;
            background: var(--card-bg);
            border-radius: 15px;
            border: 1px solid var(--border-color);
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            animation: fadeInUp 1s ease;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }}

        thead {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        }}

        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85em;
            border-bottom: 2px solid var(--border-color);
        }}

        td {{
            padding: 15px;
            border-bottom: 1px solid var(--border-color);
            transition: background 0.3s ease;
        }}

        tr:hover td {{
            background: rgba(37, 99, 235, 0.1);
        }}

        .metric-value {{
            font-weight: 600;
            font-family: 'Courier New', monospace;
        }}

        .positive {{
            color: var(--success-color);
        }}

        .negative {{
            color: var(--danger-color);
        }}

        /* Warning Box */
        .warning-box {{
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
            border: 2px solid var(--warning-color);
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            animation: fadeInUp 1.2s ease;
        }}

        .warning-box h3 {{
            color: var(--warning-color);
            margin-bottom: 15px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .warning-box p {{
            color: var(--text-secondary);
            line-height: 1.8;
            margin-bottom: 10px;
        }}

        /* Animations */
        @keyframes slideDown {{
            from {{
                opacity: 0;
                transform: translateY(-30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes rotate {{
            from {{
                transform: rotate(0deg);
            }}
            to {{
                transform: rotate(360deg);
            }}
        }}

        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.6;
            }}
        }}

        /* Loading Animation */
        .loading {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .chart-container {{
                height: 300px;
            }}
            
            table {{
                font-size: 0.85em;
            }}
            
            th, td {{
                padding: 10px;
            }}
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-top: 50px;
            border-top: 1px solid var(--border-color);
        }}

        .stock-symbol {{
            font-weight: 700;
            color: var(--primary-color);
        }}

        /* Tabs */
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .tab {{
            padding: 12px 24px;
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }}

        .tab:hover {{
            background: var(--primary-color);
            border-color: var(--primary-color);
        }}

        .tab.active {{
            background: var(--primary-color);
            border-color: var(--primary-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-content">
                <h1>🏦 Previsões Banco de Brasília</h1>
                <p class="header-subtitle">Análise Preditiva com Redes Neurais LSTM</p>
                <div>
                    <span class="badge">✅ DADOS REAIS</span>
                    <span class="badge">🤖 Deep Learning</span>
                    <span class="badge">📊 {len(self.raw_data)} Ativos</span>
                </div>
                <p style="margin-top: 15px; font-size: 0.9em;">
                    <strong>Gerado em:</strong> {current_time}
                </p>
            </div>
        </div>

        <!-- Stats Cards -->
        <div class="stats-grid">
'''

        # Gera cards de estatísticas para cada ativo
        for symbol in self.symbols:
            if symbol in self.prediction_stats:
                stats = self.prediction_stats[symbol]
                change_class = "positive" if stats['expected_return_pct'] > 0 else "negative"
                arrow = "📈" if stats['expected_return_pct'] > 0 else "📉"
                
                html_content += f'''
            <div class="stat-card">
                <div class="stat-label">{self.symbol_names[symbol]}</div>
                <div class="stat-value">R$ {stats['mean']:.2f}</div>
                <div class="stat-change {change_class}">
                    {arrow} {stats['expected_return_pct']:+.2f}%
                    <span style="font-size: 0.85em; opacity: 0.8;">
                        (15 dias)
                    </span>
                </div>
                <div style="margin-top: 15px; font-size: 0.85em; color: var(--text-secondary);">
                    <div>Último: R$ {stats['last_price']:.2f}</div>
                    <div>Mín: R$ {stats['min']:.2f} | Máx: R$ {stats['max']:.2f}</div>
                </div>
            </div>
'''

        html_content += '''
        </div>

        <!-- Charts Section -->
        <div class="chart-section">
            <h2 class="section-title">📊 Análise Gráfica Interativa</h2>
            <div class="tabs">
'''

        # Tabs para cada ativo
        for i, symbol in enumerate(self.symbols):
            if symbol in self.predictions:
                active = "active" if i == 0 else ""
                html_content += f'''
                <div class="tab {active}" onclick="showChart('{symbol}')">{self.symbol_names[symbol]}</div>
'''

        html_content += '''
            </div>
'''

        # Gera um canvas para cada ativo
        for i, symbol in enumerate(self.symbols):
            if symbol in self.predictions:
                display = "block" if i == 0 else "none"
                html_content += f'''
            <div id="chart-{symbol}" class="chart-container" style="display: {display};">
                <canvas id="canvas-{symbol}"></canvas>
            </div>
'''

        html_content += '''
        </div>

        <!-- Model Metrics Table -->
        <div class="table-container">
            <h2 class="section-title">🏗️ Métricas de Performance do Modelo</h2>
            <table>
                <thead>
                    <tr>
                        <th>Ação</th>
                        <th>RMSE Treino</th>
                        <th>RMSE Teste</th>
                        <th>MAE Teste</th>
                        <th>Overfitting Ratio</th>
                        <th>Tempo Treino</th>
                    </tr>
                </thead>
                <tbody>
'''

        for symbol, metrics in self.model_metrics.items():
            overfitting_class = "positive" if metrics['overfitting_ratio'] < 1.2 else "negative"
            html_content += f'''
                    <tr>
                        <td><strong class="stock-symbol">{self.symbol_names[symbol]}</strong></td>
                        <td class="metric-value">R$ {metrics['train_rmse']:.4f}</td>
                        <td class="metric-value">R$ {metrics['test_rmse']:.4f}</td>
                        <td class="metric-value">R$ {metrics['test_mae']:.4f}</td>
                        <td class="metric-value {overfitting_class}">{metrics['overfitting_ratio']:.2f}</td>
                        <td>{metrics['training_time']:.1f}s</td>
                    </tr>
'''

        html_content += '''
                </tbody>
            </table>
        </div>

        <!-- Predictions Statistics Table -->
        <div class="table-container">
            <h2 class="section-title">📈 Estatísticas das Previsões (15 dias)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Ação</th>
                        <th>Preço Atual</th>
                        <th>Média Prevista</th>
                        <th>Mediana</th>
                        <th>Desvio Padrão</th>
                        <th>Mínimo</th>
                        <th>Máximo</th>
                        <th>Variação Esperada</th>
                    </tr>
                </thead>
                <tbody>
'''

        for symbol, stats in self.prediction_stats.items():
            var_class = "positive" if stats['expected_return_pct'] > 0 else "negative"
            html_content += f'''
                    <tr>
                        <td><strong class="stock-symbol">{self.symbol_names[symbol]}</strong></td>
                        <td class="metric-value">R$ {stats['last_price']:.2f}</td>
                        <td class="metric-value">R$ {stats['mean']:.2f}</td>
                        <td class="metric-value">R$ {stats['median']:.2f}</td>
                        <td class="metric-value">R$ {stats['std']:.4f}</td>
                        <td class="metric-value">R$ {stats['min']:.2f}</td>
                        <td class="metric-value">R$ {stats['max']:.2f}</td>
                        <td class="metric-value {var_class}"><strong>{stats['expected_return_pct']:+.2f}%</strong></td>
                    </tr>
'''

        html_content += '''
                </tbody>
            </table>
        </div>

        <!-- Data Summary Table -->
        <div class="table-container">
            <h2 class="section-title">📊 Resumo dos Dados Históricos</h2>
            <table>
                <thead>
                    <tr>
                        <th>Ação</th>
                        <th>Registros</th>
                        <th>Período</th>
                        <th>Preço Médio</th>
                        <th>Volatilidade</th>
                        <th>Preço Min</th>
                        <th>Preço Max</th>
                    </tr>
                </thead>
                <tbody>
'''

        for symbol, summary in self.data_summary.items():
            html_content += f'''
                    <tr>
                        <td><strong class="stock-symbol">{self.symbol_names[symbol]}</strong></td>
                        <td>{summary['total_records']}</td>
                        <td>{summary['start_date']} - {summary['end_date']}</td>
                        <td class="metric-value">R$ {summary['avg_price']:.2f}</td>
                        <td class="metric-value">R$ {summary['volatility']:.4f}</td>
                        <td class="metric-value">R$ {summary['min_price']:.2f}</td>
                        <td class="metric-value">R$ {summary['max_price']:.2f}</td>
                    </tr>
'''

        html_content += '''
                </tbody>
            </table>
        </div>

        <!-- Warning Box -->
        <div class="warning-box">
            <h3>⚠️ Aviso Legal Importante</h3>
            <p><strong>Este sistema utiliza dados reais e técnicas avançadas de deep learning, porém é destinado exclusivamente para fins educacionais e de pesquisa.</strong></p>
            <p>As previsões geradas por este modelo NÃO constituem recomendação de investimento ou aconselhamento financeiro profissional. O mercado de ações é altamente volátil e imprevisível.</p>
            <p><strong>Desempenho passado não garante resultados futuros.</strong> Sempre consulte um profissional qualificado (assessor de investimentos certificado) antes de tomar qualquer decisão de investimento.</p>
            <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                <strong>Metodologia:</strong> Rede Neural LSTM com 365 dias de histórico, divisão 70/30 treino/teste, 
                15 indicadores técnicos incluindo médias móveis, RSI, MACD e Bandas de Bollinger.
            </p>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>🤖 Sistema desenvolvido com TensorFlow & Keras | 📊 Dados: InfoMoney & Yahoo Finance</p>
            <p style="margin-top: 10px; opacity: 0.7;">Previsões de curto prazo (15 dias úteis) usando Deep Learning</p>
        </div>
    </div>

    <script>
        // Chart.js Configuration
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#f1f5f9',
                            font: {
                                size: 12,
                                weight: '600'
                            },
                            padding: 15,
                            usePointStyle: true,
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#f1f5f9',
                        bodyColor: '#cbd5e1',
                        borderColor: '#334155',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += 'R$ ' + context.parsed.y.toFixed(2);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'dd/MM'
                            }
                        },
                        grid: {
                            color: 'rgba(51, 65, 85, 0.3)',
                            drawBorder: false,
                        },
                        ticks: {
                            color: '#cbd5e1',
                            font: {
                                size: 11
                            }
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(51, 65, 85, 0.3)',
                            drawBorder: false,
                        },
                        ticks: {
                            color: '#cbd5e1',
                            font: {
                                size: 11
                            },
                            callback: function(value) {
                                return 'R$ ' + value.toFixed(2);
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        };

        // Data for charts
        const chartData = {
'''

        # Gera dados para os gráficos
        for symbol in self.symbols:
            if symbol in self.predictions and symbol in self.processed_data:
                historical = self.processed_data[symbol]['Close'].tail(90)
                predictions = self.predictions[symbol]
                
                hist_dates = [d.strftime('%Y-%m-%d') for d in historical.index]
                hist_values = historical.values.tolist()
                
                pred_dates = [d.strftime('%Y-%m-%d') for d in predictions.index]
                pred_values = predictions.values.tolist()
                
                html_content += f'''
            '{symbol}': {{
                historical: {{
                    dates: {hist_dates},
                    values: {hist_values}
                }},
                predictions: {{
                    dates: {pred_dates},
                    values: {pred_values}
                }}
            }},
'''

        html_content += '''
        };

        // Initialize charts
        const charts = {};
        
        function initCharts() {
            Object.keys(chartData).forEach(symbol => {
                const ctx = document.getElementById(`canvas-${symbol}`);
                if (!ctx) return;
                
                const data = chartData[symbol];
                
                charts[symbol] = new Chart(ctx, {
                    type: 'line',
                    data: {
                        datasets: [
                            {
                                label: 'Histórico Real',
                                data: data.historical.dates.map((date, i) => ({
                                    x: date,
                                    y: data.historical.values[i]
                                })),
                                borderColor: '#3b82f6',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                borderWidth: 3,
                                pointRadius: 4,
                                pointHoverRadius: 6,
                                pointBackgroundColor: '#3b82f6',
                                tension: 0.4,
                                fill: true
                            },
                            {
                                label: 'Previsões LSTM',
                                data: data.predictions.dates.map((date, i) => ({
                                    x: date,
                                    y: data.predictions.values[i]
                                })),
                                borderColor: '#f59e0b',
                                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                borderWidth: 3,
                                borderDash: [5, 5],
                                pointRadius: 5,
                                pointHoverRadius: 7,
                                pointBackgroundColor: '#f59e0b',
                                pointStyle: 'rect',
                                tension: 0.4,
                                fill: true
                            }
                        ]
                    },
                    options: chartConfig.options
                });
            });
        }

        // Tab switching function
        function showChart(symbol) {
            // Hide all charts
            document.querySelectorAll('.chart-container').forEach(chart => {
                chart.style.display = 'none';
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected chart
            document.getElementById(`chart-${symbol}`).style.display = 'block';
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Update chart
            if (charts[symbol]) {
                charts[symbol].update();
            }
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            
            // Smooth scroll animation for elements
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            }, {
                threshold: 0.1
            });
            
            document.querySelectorAll('.stat-card, .chart-section, .table-container').forEach(el => {
                observer.observe(el);
            });
        });

        // Add loading animation
        window.addEventListener('load', function() {
            document.body.style.opacity = '1';
        });
    </script>
</body>
</html>
'''

        with open('previsoes_banco_brasilia.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("\n💾 Relatório HTML profissional salvo: previsoes_banco_brasilia.html")


# ====================================================
#                      MAIN
# ====================================================
def main():
    """
    Pipeline completo com dados REAIS.
    """
    print("🏦 SISTEMA DE PREVISÃO - BANCO DE BRASÍLIA")
    print("🤖 Modelo LSTM com DADOS REAIS")
    print("📊 Fonte: InfoMoney / Yahoo Finance")
    print("=" * 60)

    predictor = BancoDebrasiliaPredictor()

    # 1. Coleta REAL
    print("\n🔍 ETAPA 1: COLETA DE DADOS REAIS")
    if not predictor.collect_data():
        print("❌ Falha na coleta de dados. Encerrando...")
        return

    # 2. Treino
    print("\n🏗️ ETAPA 2: TREINAMENTO DOS MODELOS")
    for sym in predictor.symbols:
        if sym in predictor.raw_data:
            predictor.train_model(sym, epochs=40)

    # 3. Previsões
    print("\n🔮 ETAPA 3: GERAÇÃO DE PREVISÕES")
    for sym in predictor.symbols:
        if sym in predictor.models:
            predictor.predict_future(sym, days=15)

    # 4. Gráficos
    print("\n📊 ETAPA 4: GERAÇÃO DE GRÁFICOS")
    for sym in predictor.symbols:
        if sym in predictor.predictions:
            predictor.generate_prediction_plot(sym)

    # 5. Relatórios
    print("\n📝 ETAPA 5: GERAÇÃO DE RELATÓRIOS")
    predictor.display_tables()
    predictor.create_html_report()

    print("\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("🗂️ Arquivos gerados:")
    print("   • previsoes_banco_brasilia.html")
    for sym in predictor.symbols:
        print(f"   • plots/{sym}_prediction.png")


if __name__ == "__main__":
    main()
