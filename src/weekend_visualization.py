import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class FinanceRecommendationVisualizer:
    def __init__(self, data_path='./data/', sample_size=30000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.transactions_df = None
        self.users_df = None
        self.cards_df = None
        self.mcc_codes = None
        
    def load_data(self):
        """데이터 로드"""
        print("데이터 로딩 중...")
        
        # 거래 데이터
        self.transactions_df = pd.read_csv(f'{self.data_path}transactions_data.csv', nrows=self.sample_size)
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        self.transactions_df['amount'] = self.transactions_df['amount'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # 사용자 데이터
        self.users_df = pd.read_csv(f'{self.data_path}users_data.csv')
        self.users_df['yearly_income'] = self.users_df['yearly_income'].str.replace('$', '').str.replace(',', '').astype(float)
        self.users_df['total_debt'] = self.users_df['total_debt'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # 카드 데이터
        self.cards_df = pd.read_csv(f'{self.data_path}cards_data.csv')
        self.cards_df['credit_limit'] = self.cards_df['credit_limit'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # MCC 코드
        with open(f'{self.data_path}mcc_codes.json', 'r') as f:
            self.mcc_codes = json.load(f)
            
        # MCC 매핑
        self.transactions_df['mcc_description'] = self.transactions_df['mcc'].astype(str).map(self.mcc_codes)
        
        # 시간 특성 추가
        self.transactions_df['year'] = self.transactions_df['date'].dt.year
        self.transactions_df['month'] = self.transactions_df['date'].dt.month
        self.transactions_df['day'] = self.transactions_df['date'].dt.day
        self.transactions_df['hour'] = self.transactions_df['date'].dt.hour
        self.transactions_df['day_of_week'] = self.transactions_df['date'].dt.dayofweek
        
        print(f"거래 데이터: {self.transactions_df.shape}")
        print(f"사용자 데이터: {self.users_df.shape}")
        print(f"카드 데이터: {self.cards_df.shape}")
        
    def create_comprehensive_dashboard(self):
        """종합 대시보드 생성"""
        fig = plt.figure(figsize=(20, 24))
        
        # 1. 거래 금액 분포
        plt.subplot(4, 3, 1)
        plt.hist(self.transactions_df['amount'].clip(-1000, 1000), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Amount ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. 시간대별 거래 패턴
        plt.subplot(4, 3, 2)
        hourly_transactions = self.transactions_df.groupby('hour').size()
        plt.plot(hourly_transactions.index, hourly_transactions.values, marker='o', linewidth=2, markersize=6)
        plt.title('Hourly Transaction Pattern', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Transactions')
        plt.grid(True, alpha=0.3)
        
        # 3. 요일별 거래 패턴
        plt.subplot(4, 3, 3)
        daily_transactions = self.transactions_df.groupby('day_of_week').size()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        bars = plt.bar(range(7), daily_transactions.values, color=plt.cm.Set3(range(7)))
        plt.title('Daily Transaction Pattern', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Transactions')
        plt.xticks(range(7), day_names)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. 상위 MCC 카테고리
        plt.subplot(4, 3, 4)
        top_mcc = self.transactions_df['mcc_description'].value_counts().head(10)
        y_pos = np.arange(len(top_mcc))
        bars = plt.barh(y_pos, top_mcc.values, color=plt.cm.viridis(np.linspace(0, 1, len(top_mcc))))
        plt.title('Top 10 Merchant Categories', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Transactions')
        plt.yticks(y_pos, [desc[:25] + '...' if len(desc) > 25 else desc for desc in top_mcc.index])
        plt.grid(True, alpha=0.3, axis='x')
        
        # 5. 월별 거래 트렌드
        plt.subplot(4, 3, 5)
        monthly_stats = self.transactions_df.groupby('month').agg({
            'amount': ['count', 'sum']
        })
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax1 = plt.gca()
        color = 'tab:blue'
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Transaction Count', color=color)
        line1 = ax1.plot(monthly_stats.index, monthly_stats[('amount', 'count')], 
                        color=color, marker='o', linewidth=2, label='Count')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(months)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Total Amount ($)', color=color)
        line2 = ax2.plot(monthly_stats.index, monthly_stats[('amount', 'sum')], 
                        color=color, marker='s', linewidth=2, label='Amount')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Monthly Transaction Trends', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 6. 거래 금액 vs 빈도 (상위 카테고리)
        plt.subplot(4, 3, 6)
        category_stats = self.transactions_df.groupby('mcc_description').agg({
            'amount': ['count', 'mean']
        }).reset_index()
        category_stats.columns = ['category', 'frequency', 'avg_amount']
        category_stats = category_stats.sort_values('frequency', ascending=False).head(15)
        
        plt.scatter(category_stats['frequency'], category_stats['avg_amount'], 
                   s=100, alpha=0.6, c=range(len(category_stats)), cmap='viridis')
        plt.xlabel('Transaction Frequency')
        plt.ylabel('Average Amount ($)')
        plt.title('Category Frequency vs Average Amount', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 7. 사용자 연령 분포
        plt.subplot(4, 3, 7)
        plt.hist(self.users_df['current_age'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('User Age Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Number of Users')
        plt.grid(True, alpha=0.3)
        
        # 8. 연수입 vs 신용점수
        plt.subplot(4, 3, 8)
        plt.scatter(self.users_df['yearly_income'], self.users_df['credit_score'], 
                   alpha=0.6, s=50, color='purple')
        plt.xlabel('Yearly Income ($)')
        plt.ylabel('Credit Score')
        plt.title('Income vs Credit Score', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 9. 카드 유형 분포
        plt.subplot(4, 3, 9)
        card_types = self.cards_df['card_type'].value_counts()
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
        plt.pie(card_types.values, labels=card_types.index, autopct='%1.1f%%', 
                colors=colors[:len(card_types)], startangle=90)
        plt.title('Card Type Distribution', fontsize=14, fontweight='bold')
        
        # 10. 지역별 거래 분포 (상위 주)
        plt.subplot(4, 3, 10)
        state_transactions = self.transactions_df['merchant_state'].value_counts().head(10)
        bars = plt.bar(range(len(state_transactions)), state_transactions.values, 
                      color=plt.cm.tab10(range(len(state_transactions))))
        plt.title('Top 10 States by Transactions', fontsize=14, fontweight='bold')
        plt.xlabel('State')
        plt.ylabel('Number of Transactions')
        plt.xticks(range(len(state_transactions)), state_transactions.index, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 11. 신용 한도 분포
        plt.subplot(4, 3, 11)
        plt.hist(self.cards_df['credit_limit'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        plt.title('Credit Limit Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Credit Limit ($)')
        plt.ylabel('Number of Cards')
        plt.grid(True, alpha=0.3)
        
        # 12. 데이터 요약 정보
        plt.subplot(4, 3, 12)
        plt.axis('off')
        
        summary_text = f"""
Dataset Summary:
• Total Transactions: {len(self.transactions_df):,}
• Unique Users: {self.transactions_df['client_id'].nunique():,}
• Unique Merchants: {self.transactions_df['merchant_id'].nunique():,}
• Date Range: {self.transactions_df['date'].min().strftime('%Y-%m-%d')} to {self.transactions_df['date'].max().strftime('%Y-%m-%d')}
• Avg Transaction: ${self.transactions_df['amount'].mean():.2f}
• Total Volume: ${self.transactions_df['amount'].sum():,.0f}
• MCC Categories: {len(self.mcc_codes)}

User Demographics:
• Avg Age: {self.users_df['current_age'].mean():.1f}
• Avg Income: ${self.users_df['yearly_income'].mean():.0f}
• Avg Credit Score: {self.users_df['credit_score'].mean():.0f}

Card Statistics:
• Total Cards: {len(self.cards_df):,}
• Avg Credit Limit: ${self.cards_df['credit_limit'].mean():.0f}
"""
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Finance Recommendation System - Comprehensive Data Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig('./reports/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_recommendation_analysis(self):
        """추천 시스템 분석 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 사용자-상점 상호작용 희소성 분석
        user_merchant_matrix = self.transactions_df.pivot_table(
            index='client_id', columns='merchant_id', values='amount', 
            aggfunc='count', fill_value=0
        )
        
        sparsity = (user_merchant_matrix == 0).sum().sum() / (user_merchant_matrix.shape[0] * user_merchant_matrix.shape[1])
        
        axes[0, 0].bar(['Non-zero', 'Zero'], [1-sparsity, sparsity], color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title('User-Item Matrix Sparsity', fontweight='bold')
        axes[0, 0].set_ylabel('Proportion')
        axes[0, 0].text(0, 1-sparsity+0.05, f'{(1-sparsity)*100:.2f}%', ha='center', fontweight='bold')
        axes[0, 0].text(1, sparsity+0.05, f'{sparsity*100:.2f}%', ha='center', fontweight='bold')
        
        # 2. 사용자별 거래 횟수 분포
        user_transaction_counts = self.transactions_df['client_id'].value_counts()
        axes[0, 1].hist(user_transaction_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Distribution of Transactions per User', fontweight='bold')
        axes[0, 1].set_xlabel('Number of Transactions')
        axes[0, 1].set_ylabel('Number of Users')
        axes[0, 1].axvline(user_transaction_counts.mean(), color='red', linestyle='--', 
                          label=f'Mean: {user_transaction_counts.mean():.1f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 상점별 인기도 분포
        merchant_popularity = self.transactions_df['merchant_id'].value_counts()
        axes[0, 2].hist(merchant_popularity, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Distribution of Merchant Popularity', fontweight='bold')
        axes[0, 2].set_xlabel('Number of Transactions')
        axes[0, 2].set_ylabel('Number of Merchants')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Long Tail 분석
        merchant_popularity_sorted = merchant_popularity.sort_values(ascending=False)
        cumsum_ratio = merchant_popularity_sorted.cumsum() / merchant_popularity_sorted.sum()
        
        axes[1, 0].plot(range(len(cumsum_ratio)), cumsum_ratio, linewidth=2)
        axes[1, 0].axhline(y=0.8, color='red', linestyle='--', label='80% of transactions')
        axes[1, 0].axvline(x=len(cumsum_ratio)*0.2, color='green', linestyle='--', label='20% of merchants')
        axes[1, 0].set_title('Long Tail Distribution (80-20 Rule)', fontweight='bold')
        axes[1, 0].set_xlabel('Merchants (ranked by popularity)')
        axes[1, 0].set_ylabel('Cumulative Transaction Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 카테고리별 사용자 다양성
        category_user_diversity = self.transactions_df.groupby('mcc_description')['client_id'].nunique().sort_values(ascending=False).head(15)
        
        y_pos = np.arange(len(category_user_diversity))
        bars = axes[1, 1].barh(y_pos, category_user_diversity.values, color=plt.cm.viridis(np.linspace(0, 1, len(category_user_diversity))))
        axes[1, 1].set_title('User Diversity by Category', fontweight='bold')
        axes[1, 1].set_xlabel('Number of Unique Users')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat for cat in category_user_diversity.index])
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # 6. 추천 시스템 도전 과제 요약
        axes[1, 2].axis('off')
        
        challenges_text = f"""
Recommendation System Challenges:

Data Characteristics:
• Sparsity: {sparsity*100:.1f}%
• Users: {user_merchant_matrix.shape[0]:,}
• Items (Merchants): {user_merchant_matrix.shape[1]:,}
• Avg interactions/user: {user_transaction_counts.mean():.1f}

Cold Start Problem:
• Users with <5 transactions: {(user_transaction_counts < 5).sum():,} ({(user_transaction_counts < 5).mean()*100:.1f}%)
• Single-transaction users: {(user_transaction_counts == 1).sum():,}

Long Tail Effect:
• Top 20% merchants handle {cumsum_ratio.iloc[int(len(cumsum_ratio)*0.2)]*100:.1f}% of transactions
• {(merchant_popularity == 1).sum():,} merchants with only 1 transaction

Scalability:
• Matrix size: {user_merchant_matrix.shape[0]:,} × {user_merchant_matrix.shape[1]:,}
• Non-zero entries: {(user_merchant_matrix != 0).sum().sum():,}
"""
        
        axes[1, 2].text(0.05, 0.95, challenges_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Recommendation System Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./reports/recommendation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_model_comparison_chart(self):
        """모델 성능 비교 차트"""
        # 평가 결과 로드
        try:
            with open('./reports/evaluation_results.json', 'r') as f:
                results = json.load(f)
        except:
            print("평가 결과 파일을 찾을 수 없습니다. 먼저 day4_evaluation.py를 실행하세요.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. RMSE & MAE 비교
        models = []
        rmse_values = []
        mae_values = []
        
        for model_name, model_results in results.items():
            if 'rmse' in model_results:
                models.append(model_results['model_name'])
                rmse_values.append(model_results['rmse'])
                mae_values.append(model_results['mae'])
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.7, color='skyblue')
        bars2 = axes[0, 0].bar(x + width/2, mae_values, width, label='MAE', alpha=0.7, color='lightcoral')
        
        axes[0, 0].set_title('Model Performance: RMSE vs MAE', fontweight='bold')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Error Value')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar1, bar2 in zip(bars1, bars2):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            axes[0, 0].text(bar1.get_x() + bar1.get_width()/2., height1 + 0.001, f'{height1:.3f}', 
                           ha='center', va='bottom', fontsize=10)
            axes[0, 0].text(bar2.get_x() + bar2.get_width()/2., height2 + 0.001, f'{height2:.3f}', 
                           ha='center', va='bottom', fontsize=10)
        
        # 2. Precision@K 비교
        k_values = [5, 10, 20]
        precision_data = {}
        
        for model_name, model_results in results.items():
            if 'precision_at_k' in model_results:
                precision_data[model_results['model_name']] = [
                    model_results['precision_at_k'].get(str(k), 0) for k in k_values
                ]
        
        x = np.arange(len(k_values))
        width = 0.25
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        for i, (model, values) in enumerate(precision_data.items()):
            bars = axes[0, 1].bar(x + i*width, values, width, label=model, alpha=0.7, color=colors[i % len(colors)])
            
            # 값 표시
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{val:.3f}', 
                               ha='center', va='bottom', fontsize=9)
        
        axes[0, 1].set_title('Precision@K Comparison', fontweight='bold')
        axes[0, 1].set_xlabel('K Value')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels([f'P@{k}' for k in k_values])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Recall@K 비교
        recall_data = {}
        
        for model_name, model_results in results.items():
            if 'recall_at_k' in model_results:
                recall_data[model_results['model_name']] = [
                    model_results['recall_at_k'].get(str(k), 0) for k in k_values
                ]
        
        for i, (model, values) in enumerate(recall_data.items()):
            bars = axes[1, 0].bar(x + i*width, values, width, label=model, alpha=0.7, color=colors[i % len(colors)])
            
            # 값 표시
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{val:.3f}', 
                               ha='center', va='bottom', fontsize=9)
        
        axes[1, 0].set_title('Recall@K Comparison', fontweight='bold')
        axes[1, 0].set_xlabel('K Value')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels([f'R@{k}' for k in k_values])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. F1@K 비교
        f1_data = {}
        
        for model_name, model_results in results.items():
            if 'f1_at_k' in model_results:
                f1_data[model_results['model_name']] = [
                    model_results['f1_at_k'].get(str(k), 0) for k in k_values
                ]
        
        for i, (model, values) in enumerate(f1_data.items()):
            bars = axes[1, 1].bar(x + i*width, values, width, label=model, alpha=0.7, color=colors[i % len(colors)])
            
            # 값 표시
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{val:.3f}', 
                               ha='center', va='bottom', fontsize=9)
        
        axes[1, 1].set_title('F1@K Comparison', fontweight='bold')
        axes[1, 1].set_xlabel('K Value')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels([f'F1@{k}' for k in k_values])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Recommendation Models Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./reports/model_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # 시각화 시스템 초기화
    viz = FinanceRecommendationVisualizer(sample_size=30000)
    
    print("🎨 Weekend 프로젝트: 종합 시각화 및 리포트 생성")
    print("=" * 60)
    
    # 1. 데이터 로드
    viz.load_data()
    
    # 2. 종합 대시보드 생성
    print("\n📊 종합 대시보드 생성 중...")
    viz.create_comprehensive_dashboard()
    
    # 3. 추천 시스템 분석 시각화
    print("\n🔍 추천 시스템 분석 시각화 생성 중...")
    viz.create_recommendation_analysis()
    
    # 4. 모델 성능 비교 차트
    print("\n📈 모델 성능 비교 차트 생성 중...")
    viz.create_model_comparison_chart()
    
    print("\n✅ 모든 시각화가 ./reports/ 폴더에 저장되었습니다!")
    print("   - comprehensive_dashboard.png")
    print("   - recommendation_analysis.png") 
    print("   - model_comparison_chart.png")
    
    print("\n🎉 Weekend 프로젝트 완료!")