import pandas as pd
import json

print("=== 빠른 데이터 테스트 ===")

# 거래 데이터 샘플 로드
print("1. 거래 데이터 샘플 로드...")
transactions_df = pd.read_csv('./data/transactions_data.csv', nrows=10000)
print(f"거래 데이터 샘플: {transactions_df.shape}")
print(f"컬럼: {list(transactions_df.columns)}")
print(f"\n샘플 데이터:")
print(transactions_df.head())

# 사용자 데이터 로드
print("\n2. 사용자 데이터 로드...")
users_df = pd.read_csv('./data/users_data.csv')
print(f"사용자 데이터: {users_df.shape}")
print(users_df.head())

# 카드 데이터 로드
print("\n3. 카드 데이터 로드...")
cards_df = pd.read_csv('./data/cards_data.csv')
print(f"카드 데이터: {cards_df.shape}")
print(cards_df.head())

# MCC 코드 로드
print("\n4. MCC 코드 로드...")
with open('./data/mcc_codes.json', 'r') as f:
    mcc_codes = json.load(f)
print(f"MCC 코드: {len(mcc_codes)}개")
sample_mcc = dict(list(mcc_codes.items())[:5])
for code, description in sample_mcc.items():
    print(f"{code}: {description}")

# 기본 전처리 테스트
print("\n5. 기본 전처리 테스트...")
transactions_df['date'] = pd.to_datetime(transactions_df['date'])
transactions_df['amount'] = transactions_df['amount'].str.replace('$', '').str.replace(',', '').astype(float)
transactions_df['mcc_description'] = transactions_df['mcc'].astype(str).map(mcc_codes)

print(f"전처리 후:")
print(transactions_df[['date', 'amount', 'mcc', 'mcc_description']].head())

# 사용자-아이템 매트릭스 생성 테스트
print("\n6. 사용자-아이템 매트릭스 생성 테스트...")
user_merchant_matrix = transactions_df.groupby(['client_id', 'merchant_id'])['amount'].count().reset_index()
user_merchant_matrix.columns = ['client_id', 'merchant_id', 'transaction_count']

print(f"사용자-상점 상호작용: {user_merchant_matrix.shape}")
print(user_merchant_matrix.head())

user_mcc_matrix = transactions_df.groupby(['client_id', 'mcc'])['amount'].count().reset_index()
user_mcc_matrix.columns = ['client_id', 'mcc', 'transaction_count']

print(f"사용자-MCC 상호작용: {user_mcc_matrix.shape}")
print(user_mcc_matrix.head())

# 추천 시스템을 위한 피벗 테이블
print("\n7. 추천 시스템용 피벗 테이블...")
user_merchant_pivot = transactions_df.pivot_table(
    index='client_id',
    columns='merchant_id',
    values='amount',
    aggfunc='count',
    fill_value=0
)

print(f"사용자-상점 피벗 테이블 크기: {user_merchant_pivot.shape}")
print(f"Sparsity: {(user_merchant_pivot == 0).sum().sum() / (user_merchant_pivot.shape[0] * user_merchant_pivot.shape[1]) * 100:.2f}%")

print("\n=== Day 1 샘플 테스트 완료! ===")
print(f"고유 사용자 수: {transactions_df['client_id'].nunique()}")
print(f"고유 상점 수: {transactions_df['merchant_id'].nunique()}")
print(f"고유 MCC 카테고리 수: {transactions_df['mcc'].nunique()}")
print(f"평균 거래 금액: ${transactions_df['amount'].mean():.2f}")