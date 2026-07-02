import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MonthData:
    """Manages data ingestion, cleaning, and extraction for a single month across various formats."""
    def __init__(self, file_buffer, file_type):
        self.file_buffer = file_buffer
        self.file_type = file_type
        self.df = None
        self.error_msg = None
        self._load_file()
        if self.df is not None:
            self._clean_and_prepare()

    def _load_file(self):
        """Dynamically switches between pandas parsers and handles empty line skips."""
        try:
            # Reset buffer pointer just in case
            self.file_buffer.seek(0)
            
            if 'CSV' in self.file_type:
                # skip_blank_lines=True helps handle messy CSV structures
                self.df = pd.read_csv(self.file_buffer, skip_blank_lines=True)
            elif 'Excel' in self.file_type:
                self.df = pd.read_excel(self.file_buffer)
            elif 'JSON' in self.file_type:
                self.df = pd.read_json(self.file_buffer)
                
            # Final check to see if the dataframe actually loaded any data
            if self.df is None or self.df.empty:
                self.error_msg = "The file appears to be empty or has no readable columns."
                self.df = None
        except Exception as e:
            self.error_msg = f"Failed to parse file: {str(e)}"

    def _clean_and_prepare(self):
        """Ensures core budget columns exist and sanitizes them safely."""
        # Standardize column names to Title Case for easier matching
        self.df.columns = [str(col).strip().title() for col in self.df.columns]
        
        # Validate critical columns
        required_cols = ['Date', 'Category', 'Amount']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            self.error_msg = f"Missing required columns: {', '.join(missing)}"
            return

        try:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            self.df['Category'] = self.df['Category'].fillna('Unknown').astype(str).str.lower().str.strip()
            
            # Convert Amount safely to numeric
            self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
            mean_amount = self.df['Amount'].mean()
            self.df['Amount'] = self.df['Amount'].fillna(mean_amount if pd.notna(mean_amount) else 0.0)
            
            if 'Payment_Method' in self.df.columns:
                self.df['Payment_Method'] = self.df['Payment_Method'].fillna('Not Provided')
        except Exception as e:
            self.error_msg = f"Data normalization error: {str(e)}"

    def get_total_by_category(self):
        if self.df is None or self.error_msg:
            return pd.Series(dtype=float)
        return self.df.groupby('Category')['Amount'].sum()

    def get_grand_total(self):
        if self.df is None or self.error_msg:
            return 0.0
        return self.df['Amount'].sum()


class BudgetTracker:
    """Handles regression timeline math dynamically for N uploaded months."""
    def __init__(self, months_list):
        self.months = months_list  
        self.x_history = list(range(1, len(months_list) + 1))
        self.x_predict = len(months_list) + 1
        self.all_timeline = self.x_history + [self.x_predict]

    def get_all_unique_categories(self):
        categories = set()
        for m in self.months:
            if not m.error_msg:
                categories.update(m.get_total_by_category().index)
        return sorted(list(categories))

    def get_category_history(self, category):
        history = []
        for m in self.months:
            totals = m.get_total_by_category()
            history.append(float(totals.get(category, 0.0)))
        return history

    def calculate_regression(self, history_series):
        """Pure NumPy Linear Regression Engine (y = mx + b) adapts to any timeline length."""
        y_arr = np.array(history_series)
        x_arr = np.array(self.x_history)
        N = len(self.x_history)
        
        denominator = (N * sum(x_arr**2) - (sum(x_arr))**2)
        if denominator == 0:
            fallback_val = float(y_arr[0]) if len(y_arr) > 0 else 0.0
            return 0.0, fallback_val, fallback_val, np.full(len(self.all_timeline), fallback_val)
            
        m = (N * sum(x_arr * y_arr) - sum(x_arr) * sum(y_arr)) / denominator
        b = (sum(y_arr) - m * sum(x_arr)) / N
        
        y_pred = max(0.0, m * self.x_predict + b) 
        trend = m * np.array(self.all_timeline) + b
        trend = np.clip(trend, 0, None) 
        return m, b, y_pred, trend


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Dynamic Budget Forecaster", layout="wide")

st.title("Dynamic Budget Forecaster")
st.markdown("Predict future financial horizons using historical multi-month statements.")
st.markdown("---")

# Organized Sidebar configuration layout
with st.sidebar:
    st.header("side bar controls")
    supported_extensions = ['CSV (.csv)', 'Excel (.xlsx, .xls)', 'JSON (.json)']
    selected_extension = st.selectbox("1. Select File Format Type:", supported_extensions)

    extension_map = {
        'CSV (.csv)': ['csv'],
        'Excel (.xlsx, .xls)': ['xlsx', 'xls'],
        'JSON (.json)': ['json']
    }

    uploaded_files = st.file_uploader(
        "2. Upload Historical Files (Min 3, Max 6)", 
        type=extension_map[selected_extension], 
        accept_multiple_files=True
    )

    # State reset engine if inputs fluctuate
    if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded_files:
        st.session_state['processed'] = False
        st.session_state['last_uploaded'] = uploaded_files

    st.markdown("---")
    submit_btn = st.button("Start Predicting", use_container_width=True)

# Process verification sequence 
if uploaded_files:
    num_files = len(uploaded_files)
    if num_files < 3 or num_files > 6:
        st.error(f"404 Invalid configuration! You uploaded {num_files} files. Please upload between 3 and 6 sequential statements.")
    else:
        if submit_btn:
            st.session_state['processed'] = True
            st.session_state['files_cache'] = uploaded_files
            st.session_state['extension_cache'] = selected_extension

# Application View Render Strategy
if st.session_state.get('processed') and 'files_cache' in st.session_state:
    months_data = [MonthData(f, st.session_state['extension_cache']) for f in st.session_state['files_cache']]
    
    # Check if data normalization generated explicit faults
    errors = [f"File **{m.file_buffer.name}**: {m.error_msg}" for m in months_data if m.error_msg]
    
    if errors:
        for err in errors:
            st.error(err)
    else:
        tracker = BudgetTracker(months_data)
        base_categories = tracker.get_all_unique_categories()

        # Modern controls positioned on top of the graphics card workspace
        c1, _ = st.columns([4, 8])
        with c1:
            dropdown_options = ['total', 'all'] + base_categories
            user_choice = st.selectbox("Select Target Analytics View:", dropdown_options)

        st.markdown("### Budget Trajectory & Insights")
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plt.style.use('fast')

        # VIEW OPTION 1: COMBINED TOTAL BUDGET
        if user_choice == 'total':
            total_history = [m.get_grand_total() for m in tracker.months]
            _, _, y_pred_t, trend_t = tracker.calculate_regression(total_history)
            
            ax.plot(tracker.x_history, total_history, marker='s', color='#8a2be2', linewidth=2.5, label='Past Spending History')
            ax.plot(tracker.x_predict, y_pred_t, marker='o', color='#ff4b4b', markersize=10, label=f'Month {tracker.x_predict} Prediction')
            ax.plot(tracker.all_timeline, trend_t, color='gray', linestyle='--', alpha=0.7, label='Regression Trend Baseline')
            
            ax.set_title("Overall Monthly Spending Aggregates", fontsize=12, fontweight='bold')
            ax.legend()
            st.metric(label=f"Predicted Spending for Month {tracker.x_predict}", value=f"${y_pred_t:,.2f}")

        # VIEW OPTION 2: UNIFIED MULTI-LINE MAP
        elif user_choice == 'all':
            for cat in base_categories:
                cat_history = tracker.get_category_history(cat)
                _, _, y_pred_c, trend_c = tracker.calculate_regression(cat_history)
                
                line, = ax.plot(tracker.x_history, cat_history, marker='s', linestyle='-', linewidth=1.8, label=cat.title())
                ax.plot(tracker.x_predict, y_pred_c, marker='o', color=line.get_color(), markersize=7)
                ax.plot(tracker.all_timeline, trend_c, color=line.get_color(), linestyle=':', alpha=0.4)
            
            ax.set_title("Multi-Category Unified Trend Dashboard", fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        # VIEW OPTION 3: TARGETED INDIVIDUAL CATEGORY
        else:
            single_history = tracker.get_category_history(user_choice)
            _, _, y_pred_s, trend_s = tracker.calculate_regression(single_history)
            
            ax.plot(tracker.x_history, single_history, marker='s', color='#1f77b4', linewidth=2.5, label='Past Spending History')
            ax.plot(tracker.x_predict, y_pred_s, marker='o', color='#ff4b4b', markersize=10, label=f'Month {tracker.x_predict} Prediction')
            ax.plot(tracker.all_timeline, trend_s, color='gray', linestyle='--', alpha=0.7, label='Regression Trend Line')
            
            ax.set_title(f"Spending Trajectory for '{user_choice.title()}'", fontsize=12, fontweight='bold')
            ax.legend()
            st.metric(label=f"Predicted Month {tracker.x_predict} Spending for '{user_choice.title()}'", value=f"${y_pred_s:,.2f}")

        # Global graph aesthetic updates
        ax.set_xlabel("Sequential Month Chronology", fontsize=10)
        ax.set_ylabel("Amount Spent ($)", fontsize=10)
        ax.set_xticks(tracker.all_timeline)
        ax.grid(True, linestyle=':' if user_choice == 'all' else '-', alpha=0.6)
        
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("ℹ️ Please upload 3 to 6 historical file logs using the sidebar panel and click 'Start Predicting' to proceed.")

 
