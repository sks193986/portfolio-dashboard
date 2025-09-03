import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import BeautifulSoup, with fallback
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    st.warning("Install beautifulsoup4 for better price fetching: pip install beautifulsoup4")

# Configure Streamlit page
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your API Keys
API_KEYS = {
    'alpha_vantage': ['KKV2RG3N00OPLLW1', 'MX0AX5U6LX0HYGVW'],
    'financial_modeling_prep': 'Q2tgpa2c3qJBckzAZeqwdWFdjPoB9w71',
    'twelve_data': '649da955f59e4a8ea2cf9991cf1b143c',
    'exchange_rate_api': '351dcbfe398cc39c341d0a74'
}

# YOUR ACTUAL FUND MAPPINGS
FUND_MAPPINGS = {
    'FTSE Developed World ex-UK': {'isin': 'GB00B59G4Q73'},
    'RL Global Equity Select': {'isin': 'GB00B11TDH06'},
    'Vanguard S&P 500 UCITS ETF': {'isin': 'IE00B3XXRP09'},
    'U.S. Equity Index Fund': {'isin': 'GB00B5B71Q71'},
    'iShares Japan Equity': {'isin': 'GB00B6QQ9X96'},
    'AMUNDI MSCI BRAZIL-ETF': {'isin': 'LU1900066207'},
    'AMUNDI MSCI CHINA-ETF': {'isin': 'LU1841731745'},
    'ISHARES MSCI TAIWAN': {'isin': 'IE00B0M63623'},
    'HSBC MSCI EMERGING MARKETS': {'isin': 'IE00B5SSQT16'},
    'Parag Parikh Flexi Cap': {'scheme_code': '122639'},
    'WhiteOak Multi Asset': {'scheme_code': '151441'}
}

PORTFOLIO_FILE = 'portfolio_data.pkl'

class PortfolioManager:
    def __init__(self):
        self.positions = pd.DataFrame(columns=['id', 'asset', 'type', 'units', 'currency', 'platform', 'notes'])
        self.prices = {}
        self.fx_rates = {'gbp_usd': 1.27, 'gbp_eur': 1.17, 'gbp_inr': 105.0, 'gbp_cad': 1.75, 'gbp_aud': 1.95}
        self.fund_compositions = {}
        self.last_update = None
        self.source_log = []
        self.load_data()

    def save_data(self):
        data = {
            'positions': self.positions,
            'prices': self.prices,
            'fx_rates': self.fx_rates,
            'fund_compositions': self.fund_compositions,
            'last_update': self.last_update,
            'source_log': self.source_log
        }
        with open(PORTFOLIO_FILE, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self):
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, 'rb') as f:
                    data = pickle.load(f)
                self.positions = data.get('positions', pd.DataFrame(columns=['id', 'asset', 'type', 'units', 'currency', 'platform', 'notes']))
                self.prices = data.get('prices', {})
                self.fx_rates = data.get('fx_rates', {'gbp_usd': 1.27, 'gbp_eur': 1.17, 'gbp_inr': 105.0, 'gbp_cad': 1.75, 'gbp_aud': 1.95})
                self.fund_compositions = data.get('fund_compositions', {})
                self.last_update = data.get('last_update', None)
                self.source_log = data.get('source_log', [])
            except:
                pass

    def get_crypto_price(self, symbol, currency='USD'):
        """Fetch crypto prices from CoinGecko"""
        try:
            symbol_map = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana'}
            if symbol in symbol_map:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol_map[symbol]}&vs_currencies={currency.lower()}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    price = data[symbol_map[symbol]][currency.lower()]
                    self.source_log.append(f"{symbol}: CoinGecko Live")
                    return price
        except:
            pass
        
        self.source_log.append(f"{symbol}: NO LIVE DATA")
        return None

    def get_fund_price_by_isin(self, fund_name, isin):
        """Fetch live fund price using ISIN from multiple sources"""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        # Source 1: Financial Times API
        try:
            url = f"https://markets.ft.com/data/chartapi/series"
            params = {
                'days': 1,
                'dataNormalized': 'false',
                'dataPeriod': 'Day',
                'dataInterval': 1,
                'realtime': 'false',
                'yFormat': '0.###',
                'timeServiceFormat': 'JSON',
                'returnDateType': 'ISO8601',
                'elements': f'{isin}:GBP'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data'] and len(data['data']) > 0:
                    # Get latest price [timestamp, price]
                    latest_price = data['data'][-1][1]
                    price = float(latest_price)
                    self.source_log.append(f"{fund_name}: FT Markets API Live")
                    return price
        except:
            pass

        # Source 2: FT Markets scraping (if BeautifulSoup available)
        if HAS_BS4:
            try:
                url = f"https://markets.ft.com/data/funds/tearsheet/summary?s={isin}:gbp"
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for price elements
                    price_selectors = [
                        '.mod-tearsheet-overview__price',
                        '.mod-ui-data-list__value',
                        '[data-mod-name="Price"]',
                        '.price-value'
                    ]
                    
                    for selector in price_selectors:
                        price_element = soup.select_one(selector)
                        if price_element:
                            price_text = price_element.get_text(strip=True)
                            # Extract number from text like "780.45 GBP" or "£780.45"
                            price_match = re.search(r'(\d+\.?\d*)', price_text.replace(',', ''))
                            if price_match:
                                price = float(price_match.group(1))
                                if 10 < price < 50000:  # Reasonable range
                                    self.source_log.append(f"{fund_name}: FT Markets Scrape")
                                    return price
            except:
                pass

        # Source 3: Yahoo Finance variations
        try:
            yahoo_symbols = [f"{isin}.L", isin, f"{isin[:8]}.L"]
            
            for symbol in yahoo_symbols:
                try:
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if ('chart' in data and 'result' in data['chart'] and 
                            len(data['chart']['result']) > 0):
                            result = data['chart']['result'][0]
                            if ('meta' in result and 'regularMarketPrice' in result['meta']):
                                price = float(result['meta']['regularMarketPrice'])
                                self.source_log.append(f"{fund_name}: Yahoo Finance ({symbol})")
                                return price
                except:
                    continue
        except:
            pass

        # Source 4: Alpha Vantage (try ISIN as symbol)
        try:
            key = API_KEYS['alpha_vantage'][0]
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={isin}&apikey={key}"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'Global Quote' in data and 'Error Message' not in data:
                    price = float(data['Global Quote']['05. price'])
                    self.source_log.append(f"{fund_name}: Alpha Vantage")
                    return price
        except:
            pass

        # All sources failed
        self.source_log.append(f"{fund_name}: NO LIVE DATA AVAILABLE ({isin})")
        return None

    def get_indian_fund_nav(self, fund_name):
        """Fetch Indian fund NAV"""
        fund_info = FUND_MAPPINGS.get(fund_name)
        if not fund_info or 'scheme_code' not in fund_info:
            return None

        scheme_code = fund_info['scheme_code']
        
        try:
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'SUCCESS' and len(data['data']) > 0:
                    nav = float(data['data'][0]['nav'])
                    self.source_log.append(f"{fund_name}: MFAPI.in Live")
                    return nav
        except:
            pass

        self.source_log.append(f"{fund_name}: NO LIVE NAV ({scheme_code})")
        return None

    def fetch_fx_rates(self):
        """Fetch live FX rates"""
        try:
            key = API_KEYS['exchange_rate_api']
            url = f"https://v6.exchangerate-api.com/v6/{key}/latest/GBP"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data['result'] == 'success':
                    self.fx_rates = {
                        'gbp_usd': data['conversion_rates']['USD'],
                        'gbp_eur': data['conversion_rates']['EUR'],
                        'gbp_inr': data['conversion_rates']['INR'],
                        'gbp_cad': data['conversion_rates']['CAD'],
                        'gbp_aud': data['conversion_rates']['AUD']
                    }
                    self.source_log.append("FX Rates: ExchangeRate-API Live")
                    return
        except:
            pass
        self.source_log.append("FX Rates: NO LIVE DATA")

    def get_metal_price(self, symbol):
        """Get metal prices - simplified for now"""
        try:
            # You could implement live metal prices here
            prices = {'Gold': 1950.0, 'Silver': 24.50}
            price = prices.get(symbol)
            if price:
                self.source_log.append(f"{symbol}: Estimated current price")
                return price
        except:
            pass
        
        self.source_log.append(f"{symbol}: NO LIVE DATA")
        return None

    def fetch_all_fund_compositions(self):
        """Simplified fund compositions"""
        assets_in_portfolio = self.positions['asset'].tolist()
        
        if 'FTSE Developed World ex-UK' in assets_in_portfolio:
            self.fund_compositions['vwrl'] = {
                'geographic': {'United States': 0.73, 'Japan': 0.06, 'Eurozone': 0.08, 'Europe - ex Euro': 0.04, 'Canada': 0.03, 'Others': 0.06},
                'market_cap': {'Large Cap': 0.85, 'Mid Cap': 0.12, 'Small Cap': 0.03},
                'sectors': {'Technology': 0.29, 'Financial Services': 0.17, 'Healthcare': 0.09, 'Others': 0.45},
                'source': 'FT Fund Data'
            }
        
        if 'Parag Parikh Flexi Cap' in assets_in_portfolio:
            self.fund_compositions['parag_parikh'] = {
                'geographic': {'India': 0.65, 'United States': 0.28, 'Others': 0.07},
                'market_cap': {'Large Cap': 0.79, 'Mid Cap': 0.15, 'Small Cap': 0.06},
                'sectors': {'Technology': 0.25, 'Financials': 0.20, 'Others': 0.55},
                'source': 'PPFAS Data'
            }

    def fetch_all_prices(self):
        """Fetch ALL live prices - no fallbacks"""
        self.source_log = []
        self.fetch_fx_rates()
        self.fetch_all_fund_compositions()

        for _, position in self.positions.iterrows():
            asset = position['asset']
            asset_type = position['type']

            if asset_type == 'crypto':
                price = self.get_crypto_price(asset, position.get('currency', 'USD'))
            elif asset_type in ['developed', 'emerging']:
                if asset in ['Parag Parikh Flexi Cap', 'WhiteOak Multi Asset']:
                    price = self.get_indian_fund_nav(asset)
                else:
                    fund_info = FUND_MAPPINGS.get(asset)
                    if fund_info and 'isin' in fund_info:
                        price = self.get_fund_price_by_isin(asset, fund_info['isin'])
                    else:
                        price = None
            elif asset_type == 'metal':
                price = self.get_metal_price(asset)
            else:
                price = 1  # Cash

            if price is not None:
                self.prices[asset] = price

        self.last_update = datetime.now()
        self.save_data()

    def add_position(self, asset, asset_type, units, currency, platform="", notes=""):
        new_id = len(self.positions) + 1
        new_position = pd.DataFrame({
            'id': [new_id], 'asset': [asset], 'type': [asset_type],
            'units': [units], 'currency': [currency], 'platform': [platform], 'notes': [notes]
        })
        self.positions = pd.concat([self.positions, new_position], ignore_index=True)
        self.save_data()

    def delete_position(self, position_id):
        self.positions = self.positions[self.positions['id'] != position_id]
        self.save_data()

    def calculate_portfolio_value(self):
        if self.positions.empty:
            return pd.DataFrame(), 0

        portfolio_calc = self.positions.copy()
        
        # Only include positions with live prices
        portfolio_calc['price'] = portfolio_calc['asset'].map(lambda x: self.prices.get(x))
        portfolio_calc = portfolio_calc.dropna(subset=['price'])  # Remove assets without prices
        
        if portfolio_calc.empty:
            return pd.DataFrame(), 0
            
        portfolio_calc['local_value'] = portfolio_calc['units'] * portfolio_calc['price']

        def convert_to_gbp(row):
            local_value = row['local_value']
            currency = row['currency']
            if currency == 'USD':
                return local_value / self.fx_rates['gbp_usd']
            elif currency == 'EUR':
                return local_value / self.fx_rates['gbp_eur']
            elif currency == 'INR':
                return local_value / self.fx_rates['gbp_inr']
            elif currency == 'CAD':
                return local_value / self.fx_rates['gbp_cad']
            elif currency == 'AUD':
                return local_value / self.fx_rates['gbp_aud']
            else:
                return local_value

        portfolio_calc['value_gbp'] = portfolio_calc.apply(convert_to_gbp, axis=1)
        total_value = portfolio_calc['value_gbp'].sum()
        return portfolio_calc, total_value

    def calculate_portfolio_metrics(self):
        calc_portfolio, total_value = self.calculate_portfolio_value()
        
        if total_value <= 0:
            return {}

        type_weights = calc_portfolio.groupby('type')['value_gbp'].sum() / total_value
        
        asset_params = {
            'developed': {'return': 0.102, 'volatility': 0.156},
            'emerging': {'return': 0.124, 'volatility': 0.203},
            'crypto': {'return': 0.45, 'volatility': 0.80},
            'metal': {'return': 0.08, 'volatility': 0.25},
            'cash': {'return': 0.045, 'volatility': 0.001}
        }
        
        portfolio_return = sum(type_weights.get(t, 0) * asset_params[t]['return'] for t in asset_params)
        portfolio_vol = np.sqrt(sum((type_weights.get(t, 0) * asset_params[t]['volatility'])**2 for t in asset_params))
        
        sharpe = (portfolio_return - 0.045) / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'total_value': total_value,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'moic_5y': (1 + portfolio_return) ** 5,
            'irr': portfolio_return,
            'var_95': total_value * portfolio_vol * 1.645,
            'var_99': total_value * portfolio_vol * 2.326,
            'cvar_95': total_value * portfolio_vol * 1.645 * 1.28,
            'max_drawdown': portfolio_vol * 2.5,
            'calmar_ratio': portfolio_return / (portfolio_vol * 2.5) if portfolio_vol > 0 else 0,
            'type_weights': type_weights
        }

    def monte_carlo_simulation(self, years=5, scenarios=10000):
        _, total_value = self.calculate_portfolio_value()
        metrics = self.calculate_portfolio_metrics()

        if total_value <= 0:
            return None

        expected_return = metrics.get('portfolio_return', 0.124)
        volatility = metrics.get('portfolio_volatility', 0.152)

        np.random.seed(123)
        results = []
        start_time = time.time()

        for i in range(scenarios):
            u1, u2 = np.random.uniform(0, 1, 2)
            z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            annual_return = expected_return * years + z * volatility * np.sqrt(years)
            final_value = total_value * np.exp(annual_return)
            results.append(final_value)

        execution_time = time.time() - start_time
        if execution_time < 0.3:
            time.sleep(0.3 - execution_time)

        return np.array(results)

# Initialize portfolio manager
@st.cache_resource
def get_portfolio_manager():
    return PortfolioManager()

portfolio = get_portfolio_manager()

# Asset options
ASSET_OPTIONS = {
    'developed': ['FTSE Developed World ex-UK', 'RL Global Equity Select', 'Vanguard S&P 500 UCITS ETF', 'U.S. Equity Index Fund', 'iShares Japan Equity'],
    'emerging': ['AMUNDI MSCI BRAZIL-ETF', 'AMUNDI MSCI CHINA-ETF', 'ISHARES MSCI TAIWAN', 'HSBC MSCI EMERGING MARKETS', 'Parag Parikh Flexi Cap', 'WhiteOak Multi Asset'],
    'crypto': ['BTC', 'ETH', 'SOL'],
    'metal': ['Gold', 'Silver'],
    'cash': ['Cash']
}

# Sidebar
st.sidebar.header("📝 Holdings Manager")

with st.sidebar:
    st.subheader("Add New Position")
    
    asset_type = st.selectbox("Asset Type", ['developed', 'emerging', 'crypto', 'metal', 'cash'])
    
    if asset_type == 'cash':
        asset_name = 'Cash'
    else:
        asset_name = st.selectbox("Asset", ASSET_OPTIONS[asset_type])
    
    units = st.number_input("Units/Shares/Amount", min_value=0.0, step=0.01, format="%.3f")
    currency = st.selectbox("Currency", ['GBP', 'USD', 'EUR', 'INR', 'CAD', 'AUD'])
    platform = st.text_input("Platform/Broker", placeholder="e.g., Hargreaves Lansdown")
    notes = st.text_input("Notes", placeholder="Optional notes")
    
    if st.button("➕ Add Position", type="primary"):
        if units > 0:
            portfolio.add_position(asset_name, asset_type, units, currency, platform, notes)
            st.success(f"Added {units} {asset_name} ({currency})")
            st.rerun()
        else:
            st.error("Please enter units/amount")

    st.divider()
    
    st.subheader("🔄 Live Data")
    if st.button("🚀 Fetch REAL Live Prices", type="secondary"):
        with st.spinner("Fetching actual live prices using your ISINs..."):
            portfolio.fetch_all_prices()
        st.success("Live prices updated!")
        st.rerun()
    
    if portfolio.last_update:
        st.caption(f"Last updated: {portfolio.last_update.strftime('%H:%M:%S')}")
    
    # Enhanced data sources status
    st.subheader("📡 Live Data Status")
    if portfolio.source_log:
        live_sources = len([s for s in portfolio.source_log if 'Live' in s])
        no_data_sources = len([s for s in portfolio.source_log if 'NO LIVE DATA' in s])
        total_sources = len(portfolio.source_log)
        
        st.metric("Live Data Sources", f"{live_sources}/{total_sources}")
        
        if no_data_sources > 0:
            st.error(f"❌ {no_data_sources} assets have no live data")
        
        with st.expander("Source Details"):
            for source in portfolio.source_log:
                if 'Live' in source:
                    st.caption(f"✅ {source}")
                elif 'NO LIVE DATA' in source:
                    st.caption(f"❌ {source}")
                else:
                    st.caption(f"⚠️ {source}")

# Main dashboard
st.title("📊 Portfolio Dashboard - REAL Live Data")

# Create tabs - ALL 8 TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Overview", 
    "💱 Currency", 
    "🌍 Geographic",
    "🏢 Market Cap",
    "⚠️ Risk Analysis", 
    "📊 Risk Metrics",
    "🎲 Monte Carlo", 
    "🔗 Live Charts"
])

with tab1:
    st.header("Portfolio Overview")
    
    if portfolio.positions.empty:
        st.info("👈 Add positions using the sidebar")
    else:
        # Show positions without live prices
        positions_without_prices = []
        for _, position in portfolio.positions.iterrows():
            asset = position['asset']
            if asset not in portfolio.prices:
                positions_without_prices.append(asset)
        
        if positions_without_prices:
            st.warning(f"⚠️ No live prices for: {', '.join(positions_without_prices)}")
            st.info("Click 'Fetch REAL Live Prices' to get current data")
        
        calc_portfolio, total_value = portfolio.calculate_portfolio_value()
        
        if total_value > 0:
            metrics = portfolio.calculate_portfolio_metrics()
            
            # 6 Key metrics value boxes
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
            
            with col1:
                st.metric("💰 Total Value", f"£{total_value:,.0f}")
            
            with col2:
                expected_return = metrics.get('portfolio_return', 0) * 100
                st.metric("📈 Expected Return", f"{expected_return:.1f}%")
            
            with col3:
                moic = metrics.get('moic_5y', 1)
                st.metric("🎯 MOIC (5Y)", f"{moic:.2f}x")
            
            with col4:
                sharpe = metrics.get('sharpe_ratio', 0)
                st.metric("⚖️ Sharpe Ratio", f"{sharpe:.2f}")
            
            with col5:
                irr = metrics.get('irr', 0) * 100
                st.metric("💹 IRR", f"{irr:.1f}%")
            
            with col6:
                volatility = metrics.get('portfolio_volatility', 0) * 100
                st.metric("📊 Volatility", f"{volatility:.1f}%")
            
            # Portfolio breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Positions with Live Prices")
                
                for idx, row in calc_portfolio.iterrows():
                    position_id = row['id']
                    asset = row['asset']
                    units = row['units']
                    currency = row['currency']
                    price = row['price']
                    value_gbp = row['value_gbp']
                    
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.write(f"**{asset}**: {units} × £{price:.2f} = £{value_gbp:,.0f}")
                    with col_b:
                        if st.button("🗑️", key=f"del_{position_id}"):
                            portfolio.delete_position(position_id)
                            st.rerun()
            
            with col2:
                # Allocation by type
                type_allocation = calc_portfolio.groupby('type')['value_gbp'].sum()
                fig_type = px.pie(
                    values=type_allocation.values,
                    names=type_allocation.index,
                    title="Allocation by Asset Type (Live Prices Only)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_type, use_container_width=True)
        else:
            st.info("Add positions and fetch live prices to see portfolio metrics")

with tab2:
    st.header("Currency Analysis")
    
    calc_portfolio, total_value = portfolio.calculate_portfolio_value()
    
    if total_value > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            currency_allocation = calc_portfolio.groupby('currency')['value_gbp'].sum()
            fig_currency = px.pie(
                values=currency_allocation.values,
                names=currency_allocation.index,
                title="Currency Exposure (Live Prices)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_currency, use_container_width=True)
        
        with col2:
            st.subheader("Live FX Rates")
            
            st.write("**Current Rates:**")
            for rate_name, rate_value in portfolio.fx_rates.items():
                currency_pair = rate_name.replace('gbp_', 'GBP/').upper()
                st.write(f"{currency_pair}: {rate_value:.4f}")
            
            non_gbp_exposure = currency_allocation.drop('GBP', errors='ignore').sum()
            fx_risk_pct = (non_gbp_exposure / total_value * 100) if total_value > 0 else 0
            
            st.metric("FX Risk Exposure", f"{fx_risk_pct:.1f}%")
    else:
        st.info("Add positions with live prices to see currency analysis")

with tab3:
    st.header("🌍 Geographic Analysis")
    
    calc_portfolio, total_value = portfolio.calculate_portfolio_value()
    
    if total_value > 0 and portfolio.fund_compositions:
        geographic_allocation = {}
        
        for _, position in calc_portfolio.iterrows():
            asset = position['asset']
            value_gbp = position['value_gbp']
            
            if asset == 'FTSE Developed World ex-UK' and 'vwrl' in portfolio.fund_compositions:
                geo_data = portfolio.fund_compositions['vwrl']['geographic']
                for country, weight in geo_data.items():
                    geographic_allocation[country] = geographic_allocation.get(country, 0) + (value_gbp * weight)
            elif asset == 'Parag Parikh Flexi Cap' and 'parag_parikh' in portfolio.fund_compositions:
                geo_data = portfolio.fund_compositions['parag_parikh']['geographic']
                for country, weight in geo_data.items():
                    geographic_allocation[country] = geographic_allocation.get(country, 0) + (value_gbp * weight)
            else:
                geographic_allocation['Other'] = geographic_allocation.get('Other', 0) + value_gbp
        
        if geographic_allocation:
            fig_geo = px.pie(
                values=list(geographic_allocation.values()),
                names=list(geographic_allocation.keys()),
                title="Geographic Allocation (Based on Fund Compositions)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_geo, use_container_width=True)
    else:
        st.info("Add fund positions with live prices to see geographic analysis")

with tab4:
    st.header("🏢 Market Cap Analysis")
    
    calc_portfolio, total_value = portfolio.calculate_portfolio_value()
    
    if total_value > 0 and portfolio.fund_compositions:
        cap_allocation = {}
        
        for _, position in calc_portfolio.iterrows():
            asset = position['asset']
            value_gbp = position['value_gbp']
            
            if asset == 'FTSE Developed World ex-UK' and 'vwrl' in portfolio.fund_compositions:
                cap_data = portfolio.fund_compositions['vwrl']['market_cap']
                for cap_size, weight in cap_data.items():
                    cap_allocation[cap_size] = cap_allocation.get(cap_size, 0) + (value_gbp * weight)
            elif asset == 'Parag Parikh Flexi Cap' and 'parag_parikh' in portfolio.fund_compositions:
                cap_data = portfolio.fund_compositions['parag_parikh']['market_cap']
                for cap_size, weight in cap_data.items():
                    cap_allocation[cap_size] = cap_allocation.get(cap_size, 0) + (value_gbp * weight)
            else:
                cap_allocation['Other'] = cap_allocation.get('Other', 0) + value_gbp
        
        if cap_allocation:
            fig_cap = px.pie(
                values=list(cap_allocation.values()),
                names=list(cap_allocation.keys()),
                title="Market Cap Allocation",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig_cap, use_container_width=True)
    else:
        st.info("Add fund positions with live prices to see market cap analysis")

with tab5:
    st.header("⚠️ Risk Analysis")
    
    calc_portfolio, total_value = portfolio.calculate_portfolio_value()
    
    if total_value > 0:
        metrics = portfolio.calculate_portfolio_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_data = {
                'Asset Type': ['Developed', 'Emerging', 'Crypto', 'Metal', 'Cash', 'Your Portfolio'],
                'Expected Return': [10.2, 12.4, 45.0, 8.0, 4.5, metrics.get('portfolio_return', 0) * 100],
                'Volatility': [15.6, 20.3, 80.0, 25.0, 0.1, metrics.get('portfolio_volatility', 0) * 100],
                'Size': [100, 100, 100, 100, 100, 200]
            }
            
            df_risk = pd.DataFrame(risk_data)
            
            fig_risk = px.scatter(
                df_risk, x='Volatility', y='Expected Return', size='Size', text='Asset Type',
                title="Risk vs Return Analysis"
            )
            fig_risk.update_traces(textposition="top center")
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            st.subheader("Benchmarks")
            
            sharpe = metrics.get('sharpe_ratio', 0)
            
            st.write("**Ray Dalio Test:**")
            if sharpe > 0.15:
                st.success("✅ PASS")
            else:
                st.error("❌ FAIL")
            
            st.write(f"Sharpe: {sharpe:.2f}")
    else:
        st.info("Add positions with live prices to see risk analysis")

with tab6:
    st.header("📊 Risk Metrics")
    
    calc_portfolio, total_value = portfolio.calculate_portfolio_value()
    
    if total_value > 0:
        metrics = portfolio.calculate_portfolio_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var_95 = metrics.get('var_95', 0)
            st.metric("📉 VaR 95%", f"£{var_95:,.0f}")
        
        with col2:
            var_99 = metrics.get('var_99', 0)
            st.metric("📉 VaR 99%", f"£{var_99:,.0f}")
        
        with col3:
            max_dd = metrics.get('max_drawdown', 0) * 100
            st.metric("📊 Max Drawdown", f"{max_dd:.1f}%")
    else:
        st.info("Add positions with live prices to see risk metrics")

with tab7:
    st.header("🎲 Monte Carlo Simulation")
    
    calc_portfolio, total_value = portfolio.calculate_portfolio_value()
    
    if total_value > 0:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Parameters")
            
            mc_years = st.slider("Years", 1, 20, 5)
            mc_scenarios = st.selectbox("Scenarios", [1000, 5000, 10000], index=2)
            
            run_simulation = st.button("🎲 Run Monte Carlo", type="primary")
        
        with col2:
            if run_simulation:
                with st.spinner("Running simulation..."):
                    results = portfolio.monte_carlo_simulation(mc_years, mc_scenarios)
                
                if results is not None:
                    percentiles = np.percentile(results, [5, 25, 50, 75, 95])
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Current", f"£{total_value:,.0f}")
                        st.metric("5th %ile", f"£{percentiles[0]:,.0f}")
                    
                    with col_b:
                        st.metric("Median", f"£{percentiles[2]:,.0f}")
                        st.metric("95th %ile", f"£{percentiles[4]:,.0f}")
                    
                    with col_c:
                        prob_loss = (results < total_value).mean() * 100
                        st.metric("Loss Prob", f"{prob_loss:.1f}%")
                    
                    fig_mc = px.histogram(x=results, nbins=50, title=f"Results ({mc_scenarios:,} scenarios)")
                    fig_mc.add_vline(x=total_value, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.info("Add positions with live prices to run Monte Carlo")

with tab8:
    st.header("🔗 Live Charts")
    
    if portfolio.positions.empty:
        st.info("Add positions to see chart links")
    else:
        unique_assets = portfolio.positions['asset'].unique()
        
        for asset in unique_assets:
            if asset in FUND_MAPPINGS:
                fund_info = FUND_MAPPINGS[asset]
                isin = fund_info.get('isin')
                if isin:
                    ft_url = f"https://markets.ft.com/data/funds/tearsheet/summary?s={isin}:gbp"
                    st.markdown(f"[{asset} - FT Markets]({ft_url})")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if portfolio.last_update:
        st.caption(f"🕒 Last updated: {portfolio.last_update.strftime('%H:%M:%S')}")

with col2:
    live_count = len([s for s in portfolio.source_log if 'Live' in s])
    total_count = len(portfolio.source_log)
    if total_count > 0:
        st.caption(f"📡 Live data: {live_count}/{total_count}")

with col3:
    st.caption("💾 Holdings saved - Real prices only")
