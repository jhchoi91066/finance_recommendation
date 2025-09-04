import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class FinanceDataProcessor:
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.transactions_df = None
        self.users_df = None
        self.cards_df = None
        self.mcc_codes = None
        self.fraud_labels = None
        
    def load_data(self):
        """모든 데이터 파일을 로드합니다."""
        print("데이터 로딩을 시작합니다...")
        
        # 거래 데이터 로드
        print("거래 데이터 로딩 중...")
        self.transactions_df = pd.read_csv(f'{self.data_path}transactions_data.csv')
        print(f"거래 데이터: {self.transactions_df.shape}")
        
        # 사용자 데이터 로드
        print("사용자 데이터 로딩 중...")
        self.users_df = pd.read_csv(f'{self.data_path}users_data.csv')
        print(f"사용자 데이터: {self.users_df.shape}")
        
        # 카드 데이터 로드
        print("카드 데이터 로딩 중...")
        self.cards_df = pd.read_csv(f'{self.data_path}cards_data.csv')
        print(f"카드 데이터: {self.cards_df.shape}")
        
        # MCC 코드 로드
        print("MCC 코드 로딩 중...")
        with open(f'{self.data_path}mcc_codes.json', 'r') as f:
            self.mcc_codes = json.load(f)
        print(f"MCC 코드: {len(self.mcc_codes)}개")
        
        # 사기 라벨 로드
        print("사기 라벨 데이터 로딩 중...")
        with open(f'{self.data_path}train_fraud_labels.json', 'r') as f:
            fraud_data = json.load(f)
            self.fraud_labels = fraud_data['target']
        print(f"사기 라벨: {len(self.fraud_labels)}개")
        
        print("모든 데이터 로딩 완료!")
        
    def analyze_data_structure(self):
        """데이터 구조를 분석합니다."""
        print("\n=== 데이터 구조 분석 ===")
        
        # 거래 데이터 분석
        print("\n1. 거래 데이터 (transactions_data.csv)")
        print(f"컬럼: {list(self.transactions_df.columns)}")
        print(f"데이터 타입:\n{self.transactions_df.dtypes}")
        print(f"결측치:\n{self.transactions_df.isnull().sum()}")
        print(f"기본 통계:\n{self.transactions_df.describe()}")
        
        # 사용자 데이터 분석
        print("\n2. 사용자 데이터 (users_data.csv)")
        print(f"컬럼: {list(self.users_df.columns)}")
        print(f"데이터 타입:\n{self.users_df.dtypes}")
        print(f"결측치:\n{self.users_df.isnull().sum()}")
        
        # 카드 데이터 분석
        print("\n3. 카드 데이터 (cards_data.csv)")
        print(f"컬럼: {list(self.cards_df.columns)}")
        print(f"데이터 타입:\n{self.cards_df.dtypes}")
        print(f"결측치:\n{self.cards_df.isnull().sum()}")
        
        # MCC 코드 샘플
        print("\n4. MCC 코드 샘플")
        sample_mcc = dict(list(self.mcc_codes.items())[:5])
        for code, description in sample_mcc.items():
            print(f"{code}: {description}")
        
        # 사기 라벨 분석
        fraud_counts = pd.Series(list(self.fraud_labels.values())).value_counts()
        print(f"\n5. 사기 라벨 분포:\n{fraud_counts}")
        
    def preprocess_transactions(self):
        """거래 데이터를 전처리합니다."""
        print("\n=== 거래 데이터 전처리 ===")
        
        # 날짜 컬럼 파싱
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        
        # amount 컬럼에서 $ 제거하고 숫자로 변환
        self.transactions_df['amount'] = self.transactions_df['amount'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # 사기 라벨 매핑
        self.transactions_df['is_fraud'] = self.transactions_df['id'].astype(str).map(self.fraud_labels)
        self.transactions_df['is_fraud'] = (self.transactions_df['is_fraud'] == 'Yes').astype(int)
        
        # MCC 코드 설명 매핑
        self.transactions_df['mcc_description'] = self.transactions_df['mcc'].astype(str).map(self.mcc_codes)
        
        # 파생 변수 생성
        self.transactions_df['year'] = self.transactions_df['date'].dt.year
        self.transactions_df['month'] = self.transactions_df['date'].dt.month
        self.transactions_df['day'] = self.transactions_df['date'].dt.day
        self.transactions_df['hour'] = self.transactions_df['date'].dt.hour
        self.transactions_df['day_of_week'] = self.transactions_df['date'].dt.dayofweek
        
        # 금액 카테고리
        self.transactions_df['amount_category'] = pd.cut(
            self.transactions_df['amount'].abs(),
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        print("거래 데이터 전처리 완료!")
        print(f"전처리 후 컬럼: {list(self.transactions_df.columns)}")
        
    def create_user_item_matrix(self):
        """추천 시스템을 위한 사용자-아이템 행렬을 생성합니다."""
        print("\n=== 사용자-아이템 행렬 생성 ===")
        
        # 사용자-상점 거래 빈도 행렬
        user_merchant_matrix = self.transactions_df.groupby(['client_id', 'merchant_id'])['amount'].agg(['count', 'sum']).reset_index()
        user_merchant_matrix.columns = ['client_id', 'merchant_id', 'transaction_count', 'total_amount']
        
        # 사용자-MCC 카테고리 선호도 행렬
        user_mcc_matrix = self.transactions_df.groupby(['client_id', 'mcc'])['amount'].agg(['count', 'sum']).reset_index()
        user_mcc_matrix.columns = ['client_id', 'mcc', 'transaction_count', 'total_amount']
        
        # 피벗 테이블 생성 (추천 시스템용)
        user_merchant_pivot = self.transactions_df.pivot_table(
            index='client_id',
            columns='merchant_id',
            values='amount',
            aggfunc='count',
            fill_value=0
        )
        
        user_mcc_pivot = self.transactions_df.pivot_table(
            index='client_id',
            columns='mcc',
            values='amount',
            aggfunc='count',
            fill_value=0
        )
        
        print(f"사용자-상점 행렬 크기: {user_merchant_pivot.shape}")
        print(f"사용자-MCC 행렬 크기: {user_mcc_pivot.shape}")
        
        return user_merchant_pivot, user_mcc_pivot, user_merchant_matrix, user_mcc_matrix
        
    def generate_eda_visualizations(self):
        """탐색적 데이터 분석을 위한 시각화를 생성합니다."""
        print("\n=== EDA 시각화 생성 ===")
        
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 거래 금액 분포
        axes[0, 0].hist(self.transactions_df['amount'].clip(-1000, 1000), bins=50, alpha=0.7)
        axes[0, 0].set_title('Transaction Amount Distribution')
        axes[0, 0].set_xlabel('Amount ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 시간별 거래 패턴
        hourly_transactions = self.transactions_df.groupby('hour').size()
        axes[0, 1].plot(hourly_transactions.index, hourly_transactions.values)
        axes[0, 1].set_title('Hourly Transaction Pattern')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Number of Transactions')
        
        # 3. 요일별 거래 패턴
        daily_transactions = self.transactions_df.groupby('day_of_week').size()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 2].bar(range(7), daily_transactions.values)
        axes[0, 2].set_title('Daily Transaction Pattern')
        axes[0, 2].set_xlabel('Day of Week')
        axes[0, 2].set_ylabel('Number of Transactions')
        axes[0, 2].set_xticks(range(7))
        axes[0, 2].set_xticklabels(day_names)
        
        # 4. 상위 MCC 카테고리
        top_mcc = self.transactions_df['mcc_description'].value_counts().head(10)
        axes[1, 0].barh(range(len(top_mcc)), top_mcc.values)
        axes[1, 0].set_title('Top 10 Merchant Categories')
        axes[1, 0].set_xlabel('Number of Transactions')
        axes[1, 0].set_yticks(range(len(top_mcc)))
        axes[1, 0].set_yticklabels([desc[:20] + '...' if len(desc) > 20 else desc for desc in top_mcc.index])
        
        # 5. 사기 거래 분포
        fraud_distribution = self.transactions_df['is_fraud'].value_counts()
        axes[1, 1].pie(fraud_distribution.values, labels=['Normal', 'Fraud'], autopct='%1.1f%%')
        axes[1, 1].set_title('Fraud Transaction Distribution')
        
        # 6. 금액 카테고리별 분포
        amount_cat_counts = self.transactions_df['amount_category'].value_counts()
        axes[1, 2].bar(amount_cat_counts.index, amount_cat_counts.values)
        axes[1, 2].set_title('Amount Category Distribution')
        axes[1, 2].set_xlabel('Amount Category')
        axes[1, 2].set_ylabel('Number of Transactions')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('./reports/day1_eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("EDA 시각화 완료! 결과는 ../reports/day1_eda_analysis.png에 저장되었습니다.")
        
    def save_preprocessed_data(self):
        """전처리된 데이터를 저장합니다."""
        print("\n=== 전처리된 데이터 저장 ===")
        
        # 전처리된 거래 데이터 저장
        self.transactions_df.to_csv(f'{self.data_path}preprocessed_transactions.csv', index=False)
        print("전처리된 거래 데이터가 저장되었습니다.")
        
        # 사용자-아이템 행렬들 저장
        user_merchant_pivot, user_mcc_pivot, user_merchant_matrix, user_mcc_matrix = self.create_user_item_matrix()
        
        user_merchant_pivot.to_csv(f'{self.data_path}user_merchant_matrix.csv')
        user_mcc_pivot.to_csv(f'{self.data_path}user_mcc_matrix.csv')
        user_merchant_matrix.to_csv(f'{self.data_path}user_merchant_interactions.csv', index=False)
        user_mcc_matrix.to_csv(f'{self.data_path}user_mcc_interactions.csv', index=False)
        
        print("사용자-아이템 행렬들이 저장되었습니다.")
        
    def print_summary(self):
        """데이터 처리 요약을 출력합니다."""
        print("\n=== 데이터 처리 요약 ===")
        print(f"총 거래 수: {len(self.transactions_df):,}")
        print(f"고유 사용자 수: {self.transactions_df['client_id'].nunique():,}")
        print(f"고유 상점 수: {self.transactions_df['merchant_id'].nunique():,}")
        print(f"고유 MCC 카테고리 수: {self.transactions_df['mcc'].nunique():,}")
        print(f"사기 거래 수: {self.transactions_df['is_fraud'].sum():,}")
        print(f"사기 거래 비율: {self.transactions_df['is_fraud'].mean()*100:.2f}%")
        print(f"평균 거래 금액: ${self.transactions_df['amount'].mean():.2f}")
        print(f"거래 데이터 기간: {self.transactions_df['date'].min()} ~ {self.transactions_df['date'].max()}")

if __name__ == "__main__":
    # 데이터 처리기 초기화
    processor = FinanceDataProcessor()
    
    # 1. 데이터 로딩
    processor.load_data()
    
    # 2. 데이터 구조 분석
    processor.analyze_data_structure()
    
    # 3. 데이터 전처리
    processor.preprocess_transactions()
    
    # 4. EDA 시각화 (스킵)
    # processor.generate_eda_visualizations()
    
    # 5. 전처리된 데이터 저장
    processor.save_preprocessed_data()
    
    # 6. 요약 출력
    processor.print_summary()
    
    print("\nDay 1 데이터 전처리가 완료되었습니다!")