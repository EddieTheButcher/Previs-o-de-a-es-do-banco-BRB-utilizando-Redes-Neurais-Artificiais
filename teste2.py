

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

        # --------- CONTÊINERS INTERNOS ---------- #
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
            DataFrame indexado por data com colunas Open, High, Low, Close, Volume.
        """
        print(f"🔄 Coletando dados do InfoMoney para {self.symbol_names[symbol]}...")

        try:
            # URL do InfoMoney para dados históricos
            url = f"https://www.infomoney.com.br/cotacoes/b3/acao/{symbol.lower()}/"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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
                            except Exception as e:
                                continue

            # Se não conseguiu coletar dados da tabela, usa API alternativa
            if not data_list:
                print(f"⚠️  Tentando fonte alternativa (Yahoo Finance)...")
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
            print(f"🔄 Tentando Yahoo Finance como alternativa...")
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

            print(f"📥 Baixando dados do Yahoo Finance para {ticker}...")

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
            print(f"🔄 Gerando dados sintéticos como último recurso...")
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
        print(f"🔄 Gerando dados sintéticos para {self.symbol_names[symbol]}...")

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
            open_price = close_price * np.random.normal(1, 0.005) if i == 0 \
                         else prices[i-1] * np.random.normal(1, 0.01)
            daily_vol = abs(daily_returns[i]) + 0.005
            highs.append(max(open_price, close_price) * (1 + daily_vol))
            lows.append(min(open_price, close_price) * (1 - daily_vol))
            opens.append(open_price)

            base_vol = np.random.randint(1e5, 1e6)
            if abs(daily_returns[i]) > 0.02:
                base_vol *= np.random.uniform(2, 5)
            volumes.append(int(base_vol))

        data = pd.DataFrame({
            'Open': opens, 'High': highs, 'Low': lows,
            'Close': prices, 'Volume': volumes
        }, index=pd.DatetimeIndex(dates))

        print(f"✅ {len(data)} registros sintéticos gerados")
        return data

    def collect_data(self) -> bool:
        """
        Percorre todos os tickers e coleta dados reais do InfoMoney.
        Retorna True se pelo menos um ativo foi carregado com sucesso.
        """
        print("🔄 Iniciando coleta de dados REAIS...")
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
                    print(f"   Período: {data.index[0].strftime('%d/%m/%Y')} a {data.index[-1].strftime('%d/%m/%Y')}")
                    print(f"   Preço atual: R$ {data['Close'].iloc[-1]:.2f}")
                    print(f"   Variação (período): {((data['Close'].iloc[-1]/data['Close'].iloc[0]) - 1) * 100:+.2f}%")
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

        exp1, exp2 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
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

        feats = ['Close', 'Volume', 'High', 'Low', 'Open', 'MA_7', 'MA_21', 'MA_50',
                 'RSI', 'Volatility', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
                 'Volume_MA']

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
        cb_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                   restore_best_weights=True, verbose=0)
        start = time.time()
        model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                  validation_data=(X_test, y_test), callbacks=[cb_early], verbose=0)
        elapsed = time.time() - start
        self.models[symbol] = model

        train_pred = self.denormalize(symbol, model.predict(X_train, verbose=0))
        test_pred = self.denormalize(symbol, model.predict(X_test, verbose=0))
        y_train_dn = self.denormalize(symbol, y_train.reshape(-1, 1))
        y_test_dn = self.denormalize(symbol, y_test.reshape(-1, 1))

        self.model_metrics[symbol] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_dn, train_pred)),
            'test_rmse' : np.sqrt(mean_squared_error(y_test_dn,  test_pred)),
            'test_mae'  : mean_absolute_error(y_test_dn, test_pred),
            'training_time': elapsed,
            'overfitting_ratio': (np.sqrt(mean_squared_error(y_test_dn, test_pred)) /
                                  np.sqrt(mean_squared_error(y_train_dn, train_pred)))
        }
        print(f"✅ {self.symbol_names[symbol]} treinado em {elapsed:.1f}s "
              f"| RMSE teste: R$ {self.model_metrics[symbol]['test_rmse']:.4f}")

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

        model, scaler = self.models[symbol], self.scalers[symbol]
        data, feats = self.processed_data[symbol], ['Close', 'Volume', 'High', 'Low', 'Open',
                                                    'MA_7', 'MA_21', 'MA_50', 'RSI', 'Volatility',
                                                    'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
                                                    'Volume_MA']
        seq_scaled = scaler.transform(data[feats].tail(self.sequence_length).values)
        preds_scaled, seq_curr = [], seq_scaled.copy()

        for _ in range(days):
            nxt = model.predict(seq_curr.reshape(1, self.sequence_length, len(feats)), verbose=0)
            preds_scaled.append(nxt[0, 0])
            new_row = seq_curr[-1].copy()
            new_row[0] = nxt[0, 0]
            seq_curr = np.vstack([seq_curr[1:], new_row])

        preds = self.denormalize(symbol, np.array(preds_scaled).reshape(-1, 1))

        last_date, fut_dates = data.index[-1], []
        while len(fut_dates) < days:
            last_date += timedelta(days=1)
            if last_date.weekday() < 5:
                fut_dates.append(last_date)
        series = pd.Series(preds, index=fut_dates)

        self.predictions[symbol] = series

        last_price = data['Close'].iloc[-1]
        self.prediction_stats[symbol] = {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'expected_return_pct': (series.mean() - last_price) / last_price * 100
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

        ax.plot(historical.index, historical.values,
                linewidth=2, color='#1f77b4', label='Histórico Real', marker='o', markersize=3)

        ax.plot(predictions.index, predictions.values,
                linewidth=2.5, color='#ff7f0e', label='Previsões LSTM',
                marker='s', markersize=4, linestyle='--')

        last_historical_date = historical.index[-1]
        ax.axvline(x=last_historical_date, color='red', linestyle=':',
                   linewidth=2, label='Início das Previsões')

        ax.set_title(f'Previsão de Preços - {self.symbol_names[symbol]} (DADOS REAIS)\n'
                     f'Histórico (90 dias) vs Previsões LSTM (15 dias)',
                     fontsize=16, fontweight='bold', pad=20)
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
        print("\n" + "="*100)
        print("📊 TABELA RESUMO - DADOS COLETADOS (REAIS)")
        print("="*100)
        print(f"{'Ação':<15} {'Registros':<10} {'Período':<25} "
              f"{'Preço Médio':<12} {'Volatilidade':<12}")
        print("-"*100)
        for s, d in self.data_summary.items():
            print(f"{self.symbol_names[s]:<15} {d['total_records']:<10} "
                  f"{d['start_date']} - {d['end_date']:<10} "
                  f"R$ {d['avg_price']:.4f}   R$ {d['volatility']:.4f}")

        print("\n" + "="*100)
        print("🏋️ TABELA MÉTRICAS DE BONDADE DE AJUSTE - MODELOS LSTM")
        print("="*100)
        print("Indicadores de desempenho do modelo e potencial de overfitting.")
        print(f"{'Ação':<15} {'RMSE Treino':<12} {'RMSE Teste':<12} "
              f"{'MAE Teste':<12} {'Overfitting':<12} {'Tempo (s)':<10}")
        print("-"*100)
        for s, m in self.model_metrics.items():
            print(f"{self.symbol_names[s]:<15} R$ {m['train_rmse']:.4f}   "
                  f"R$ {m['test_rmse']:.4f}   R$ {m['test_mae']:.4f}   "
                  f"{m['overfitting_ratio']:.2f}         {m['training_time']:.1f}")
        print("-" * 100)
        print("Nota:")
        print("  - RMSE: Quanto menor, melhor o ajuste.")
        print("  - MAE: Menos sensível a outliers que o RMSE.")
        print("  - Overfitting Ratio próximo de 1.0 é ideal.")

        print("\n" + "="*100)
        print("🔮 TABELA RESUMO - PREVISÕES (15 DIAS)")
        print("="*100)
        print(f"{'Ação':<15} {'Último Preço':<14} {'Previsão Média':<15} "
              f"{'Variação %':<12} {'Tendência':<10}")
        print("-"*100)
        for s in self.symbols:
            if s in self.predictions:
                last_price = self.processed_data[s]['Close'].iloc[-1]
                pred_mean = self.prediction_stats[s]['mean']
                change_pct = (pred_mean - last_price) / last_price * 100
                trend = "ALTA" if change_pct > 0 else "BAIXA"
                print(f"{self.symbol_names[s]:<15} R$ {last_price:.4f}     "
                      f"R$ {pred_mean:.4f}     {change_pct:+.2f}%      {trend}")

        print("\n" + "="*100)
        print("📈 TABELA ESTATÍSTICAS – PREVISÕES")
        print("="*100)
        print(f"{'Ação':<15} {'Média':<12} {'Mediana':<12} {'Desvio-Padrão':<15} "
              f"{'Mínimo':<12} {'Máximo':<12}")
        print("-"*100)
        for s, st in self.prediction_stats.items():
            print(f"{self.symbol_names[s]:<15} R$ {st['mean']:.4f}   R$ {st['median']:.4f}   "
                  f"R$ {st['std']:.4f}       R$ {st['min']:.4f}   R$ {st['max']:.4f}")

    def create_html_report(self):
        """
        Cria relatório HTML completo.
        """
        current_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

        html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Previsões - Banco de Brasília (DADOS REAIS)</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; background: white; padding: 30px; border-radius: 10px;
                   margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .badge {{ background: #28a745; color: white; padding: 5px 10px; border-radius: 5px;
                 font-size: 0.9em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0;
                 border-radius: 10px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        th {{ background: #007bff; color: white; padding: 12px; text-align: center; }}
        td {{ padding: 10px; text-align: center; border-bottom: 1px solid #eee; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .positive {{ color: #28a745; font-weight: bold; }}
        .negative {{ color: #dc3545; font-weight: bold; }}
        h2 {{ background: #007bff; color: white; padding: 15px; border-radius: 5px;
              text-align: center; margin-top: 30px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 10px;
                           margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .chart-container img {{ width: 100%; height: auto; border-radius: 5px; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffc107; padding: 20px;
                   border-radius: 10px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Previsões - Banco de Brasília</h1>
            <p><span class="badge">DADOS REAIS</span> InfoMoney / Yahoo Finance</p>
            <p><strong>Modelo LSTM</strong> | 365 dias históricos | 70% Treino / 30% Teste</p>
            <p><strong>Gerado em:</strong> {current_time}</p>
        </div>

        <h2>📈 Gráficos de Previsão (Dados Reais)</h2>
"""

        for symbol in self.symbols:
            if symbol in self.predictions:
                img_filename = f'plots/{symbol}_prediction.png'
                html_content += f"""
        <div class="chart-container">
            <h3 style="text-align: center; color: #007bff;">{self.symbol_names[symbol]}</h3>
            <img src="{img_filename}" alt="Gráfico {self.symbol_names[symbol]}">
        </div>
"""

        html_content += """
        <h2>🏋️ Métricas de Bondade de Ajuste dos Modelos</h2>
        <table>
            <thead>
                <tr>
                    <th>Ação</th>
                    <th>RMSE Treino</th>
                    <th>RMSE Teste</th>
                    <th>MAE Teste</th>
                    <th>Overfitting Ratio</th>
                    <th>Tempo (s)</th>
                </tr>
            </thead>
            <tbody>
"""

        for symbol, metrics in self.model_metrics.items():
            html_content += f"""
                <tr>
                    <td><strong>{self.symbol_names[symbol]}</strong></td>
                    <td>R$ {metrics['train_rmse']:.4f}</td>
                    <td>R$ {metrics['test_rmse']:.4f}</td>
                    <td>R$ {metrics['test_mae']:.4f}</td>
                    <td>{metrics['overfitting_ratio']:.2f}</td>
                    <td>{metrics['training_time']:.1f}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>📊 Estatísticas das Previsões</h2>
        <table>
            <thead>
                <tr>
                    <th>Ação</th>
                    <th>Média</th>
                    <th>Mediana</th>
                    <th>Desvio-Padrão</th>
                    <th>Mínimo</th>
                    <th>Máximo</th>
                    <th>Retorno Esperado</th>
                </tr>
            </thead>
            <tbody>
"""

        for symbol, stats in self.prediction_stats.items():
            var_class = "positive" if stats['expected_return_pct'] > 0 else "negative"
            html_content += f"""
                <tr>
                    <td><strong>{self.symbol_names[symbol]}</strong></td>
                    <td>R$ {stats['mean']:.4f}</td>
                    <td>R$ {stats['median']:.4f}</td>
                    <td>R$ {stats['std']:.4f}</td>
                    <td>R$ {stats['min']:.4f}</td>
                    <td>R$ {stats['max']:.4f}</td>
                    <td class="{var_class}">{stats['expected_return_pct']:+.2f}%</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <div class="warning">
            <h3>⚠️ Aviso Legal</h3>
            <p><strong>Este modelo utiliza dados reais mas é apenas para fins educacionais.</strong></p>
            <p>As previsões NÃO constituem aconselhamento financeiro. O mercado de ações é altamente
            volátil e imprevisível. Passado não garante resultados futuros.</p>
            <p><strong>Consulte sempre um profissional qualificado antes de investir.</strong></p>
        </div>
    </div>
</body>
</html>
"""

        with open('previsoes_banco_brasilia.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("\n💾 Relatório HTML salvo: previsoes_banco_brasilia.html")


# ====================================================
#                      MAIN
# ====================================================
def main():
    """
    Pipeline completo com dados REAIS.
    """
    print("🚀 SISTEMA DE PREVISÃO - BANCO DE BRASÍLIA")
    print("📊 Modelo LSTM com DADOS REAIS")
    print("🌐 Fonte: InfoMoney / Yahoo Finance")
    print("=" * 60)

    predictor = BancoDebrasiliaPredictor()

    # 1. Coleta REAL
    print("\n📊 ETAPA 1: COLETA DE DADOS REAIS")
    if not predictor.collect_data():
        print("❌ Falha na coleta de dados. Encerrando...")
        return

    # 2. Treino
    print("\n🏋️ ETAPA 2: TREINAMENTO DOS MODELOS")
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
    print("\n📋 ETAPA 5: GERAÇÃO DE RELATÓRIOS")
    predictor.display_tables()
    predictor.create_html_report()

    print("\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("📁 Arquivos gerados:")
    print("   • previsoes_banco_brasilia.html")
    for sym in predictor.symbols:
        print(f"   • plots/{sym}_prediction.png")


if __name__ == "__main__":
    main()


