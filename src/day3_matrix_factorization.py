import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import json
import warnings
warnings.filterwarnings('ignore')

class MatrixFactorizationRecommender:
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.transactions_df = None
        self.user_item_matrix = None
        self.user_encoder = None
        self.item_encoder = None
        self.svd_model = None
        self.nmf_model = None
        self.train_matrix = None
        self.test_matrix = None
        
    def load_and_prepare_data(self, sample_size=50000):
        """데이터 로드 및 준비"""
        print(f"데이터 로딩 중... (샘플 크기: {sample_size})")
        
        # 거래 데이터 로드
        self.transactions_df = pd.read_csv(f'{self.data_path}transactions_data.csv', nrows=sample_size)
        
        # 기본 전처리
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        self.transactions_df['amount'] = self.transactions_df['amount'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # 음수 금액을 절댓값으로 변환 (환불 등)
        self.transactions_df['amount'] = self.transactions_df['amount'].abs()
        
        print(f"로드된 데이터: {self.transactions_df.shape}")
        print(f"고유 사용자: {self.transactions_df['client_id'].nunique()}")
        print(f"고유 상점: {self.transactions_df['merchant_id'].nunique()}")
        
    def create_rating_matrix(self):
        """사용자-상점 평점 행렬 생성"""
        print("\n=== 평점 행렬 생성 ===")
        
        # 사용자별 상점별 거래 빈도와 평균 금액으로 평점 생성
        user_item_stats = self.transactions_df.groupby(['client_id', 'merchant_id']).agg({
            'amount': ['count', 'mean', 'sum']
        }).reset_index()
        
        user_item_stats.columns = ['client_id', 'merchant_id', 'frequency', 'avg_amount', 'total_amount']
        
        # 평점을 빈도와 평균 금액의 조합으로 계산
        # 정규화: 빈도 * log(평균금액 + 1) / 최대값으로 1-5 스케일 변환
        user_item_stats['rating'] = user_item_stats['frequency'] * np.log1p(user_item_stats['avg_amount'])
        
        # 1-5 스케일로 정규화
        max_rating = user_item_stats['rating'].max()
        user_item_stats['rating'] = 1 + 4 * (user_item_stats['rating'] / max_rating)
        
        print(f"평점 통계:")
        print(f"  최소 평점: {user_item_stats['rating'].min():.2f}")
        print(f"  최대 평점: {user_item_stats['rating'].max():.2f}")
        print(f"  평균 평점: {user_item_stats['rating'].mean():.2f}")
        
        # 사용자와 아이템 ID 인코딩
        unique_users = sorted(user_item_stats['client_id'].unique())
        unique_items = sorted(user_item_stats['merchant_id'].unique())
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        self.user_encoder = user_to_idx
        self.item_encoder = item_to_idx
        
        # 인덱스 매핑
        user_item_stats['user_idx'] = user_item_stats['client_id'].map(user_to_idx)
        user_item_stats['item_idx'] = user_item_stats['merchant_id'].map(item_to_idx)
        
        # 평점 행렬 생성
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in user_item_stats.iterrows():
            self.user_item_matrix[int(row['user_idx']), int(row['item_idx'])] = row['rating']
        
        print(f"평점 행렬 크기: {self.user_item_matrix.shape}")
        print(f"Sparsity: {(self.user_item_matrix == 0).sum() / self.user_item_matrix.size * 100:.2f}%")
        
        return user_item_stats
        
    def train_test_split_matrix(self, test_ratio=0.2):
        """Train/Test 분할"""
        print(f"\n=== Train/Test 분할 (테스트 비율: {test_ratio}) ===")
        
        # 0이 아닌 평점들의 인덱스
        non_zero_indices = np.nonzero(self.user_item_matrix)
        non_zero_data = list(zip(non_zero_indices[0], non_zero_indices[1], 
                                self.user_item_matrix[non_zero_indices]))
        
        # Train/Test 분할
        train_data, test_data = train_test_split(non_zero_data, test_size=test_ratio, random_state=42)
        
        # Train 행렬 생성
        self.train_matrix = np.zeros_like(self.user_item_matrix)
        for user_idx, item_idx, rating in train_data:
            self.train_matrix[user_idx, item_idx] = rating
        
        # Test 행렬 생성
        self.test_matrix = np.zeros_like(self.user_item_matrix)
        for user_idx, item_idx, rating in test_data:
            self.test_matrix[user_idx, item_idx] = rating
        
        print(f"Train 데이터: {len(train_data)}개 평점")
        print(f"Test 데이터: {len(test_data)}개 평점")
        
        return train_data, test_data
        
    def train_svd_model(self, n_components=50):
        """SVD 모델 학습"""
        print(f"\n=== SVD 모델 학습 (컴포넌트: {n_components}) ===")
        
        # SVD 모델 초기화
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # 학습
        self.svd_model.fit(self.train_matrix)
        
        # 차원 축소된 사용자 특성과 아이템 특성
        user_features = self.svd_model.transform(self.train_matrix)
        item_features = self.svd_model.components_
        
        print(f"사용자 특성 행렬 크기: {user_features.shape}")
        print(f"아이템 특성 행렬 크기: {item_features.shape}")
        print(f"설명된 분산 비율: {self.svd_model.explained_variance_ratio_.sum():.4f}")
        
        return user_features, item_features
        
    def train_nmf_model(self, n_components=50):
        """NMF 모델 학습"""
        print(f"\n=== NMF 모델 학습 (컴포넌트: {n_components}) ===")
        
        # NMF 모델 초기화 (음수 값이 없어야 함)
        self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=500)
        
        # 학습
        user_features = self.nmf_model.fit_transform(self.train_matrix)
        item_features = self.nmf_model.components_
        
        print(f"사용자 특성 행렬 크기: {user_features.shape}")
        print(f"아이템 특성 행렬 크기: {item_features.shape}")
        print(f"재구성 에러: {self.nmf_model.reconstruction_err_:.4f}")
        
        return user_features, item_features
        
    def predict_ratings(self, model_type='svd'):
        """평점 예측"""
        print(f"\n=== {model_type.upper()} 평점 예측 ===")
        
        if model_type == 'svd' and self.svd_model:
            predicted_matrix = self.svd_model.transform(self.train_matrix) @ self.svd_model.components_
        elif model_type == 'nmf' and self.nmf_model:
            predicted_matrix = self.nmf_model.transform(self.train_matrix) @ self.nmf_model.components_
        else:
            print(f"{model_type} 모델이 학습되지 않았습니다.")
            return None
        
        # 예측값을 1-5 범위로 클리핑
        predicted_matrix = np.clip(predicted_matrix, 1, 5)
        
        return predicted_matrix
        
    def evaluate_model(self, predicted_matrix, model_name):
        """모델 성능 평가"""
        print(f"\n=== {model_name} 모델 평가 ===")
        
        # RMSE 계산 (테스트 데이터에 대해서만)
        test_indices = np.nonzero(self.test_matrix)
        actual_ratings = self.test_matrix[test_indices]
        predicted_ratings = predicted_matrix[test_indices]
        
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        mae = np.mean(np.abs(actual_ratings - predicted_ratings))
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # 평점 분포 분석
        print(f"실제 평점 분포:")
        print(f"  평균: {actual_ratings.mean():.2f}")
        print(f"  표준편차: {actual_ratings.std():.2f}")
        print(f"예측 평점 분포:")
        print(f"  평균: {predicted_ratings.mean():.2f}")
        print(f"  표준편차: {predicted_ratings.std():.2f}")
        
        return rmse, mae
        
    def get_recommendations(self, user_id, model_type='svd', n_recommendations=10):
        """사용자에 대한 추천 생성"""
        print(f"\n=== 사용자 {user_id}에 대한 {model_type.upper()} 추천 ===")
        
        if user_id not in self.user_encoder:
            print(f"사용자 {user_id}가 존재하지 않습니다.")
            return []
        
        user_idx = self.user_encoder[user_id]
        
        # 예측 평점 행렬 가져오기
        predicted_matrix = self.predict_ratings(model_type)
        if predicted_matrix is None:
            return []
        
        # 사용자의 예측 평점
        user_predictions = predicted_matrix[user_idx]
        
        # 이미 평점을 준 아이템들 제외
        rated_items = np.nonzero(self.user_item_matrix[user_idx])[0]
        
        # 추천할 아이템들 (평점을 주지 않은 아이템들)
        unrated_items = np.setdiff1d(np.arange(len(user_predictions)), rated_items)
        
        if len(unrated_items) == 0:
            print("추천할 수 있는 새로운 상점이 없습니다.")
            return []
        
        # 예측 평점 기준으로 정렬
        unrated_predictions = user_predictions[unrated_items]
        top_indices = np.argsort(unrated_predictions)[::-1][:n_recommendations]
        top_items = unrated_items[top_indices]
        top_ratings = unrated_predictions[top_indices]
        
        # 실제 아이템 ID로 변환
        idx_to_item = {idx: item for item, idx in self.item_encoder.items()}
        recommendations = []
        
        print(f"추천 상점 (상위 {n_recommendations}개):")
        for i, (item_idx, rating) in enumerate(zip(top_items, top_ratings), 1):
            merchant_id = idx_to_item[item_idx]
            
            # 상점 정보 가져오기
            merchant_info = self.transactions_df[self.transactions_df['merchant_id'] == merchant_id].iloc[0]
            
            print(f"  {i}. 상점ID: {merchant_id}, 예측평점: {rating:.2f}")
            print(f"     위치: {merchant_info['merchant_city']}, {merchant_info['merchant_state']}")
            try:
                with open(f'{self.data_path}mcc_codes.json', 'r') as f:
                    mcc_codes = json.load(f)
                category = mcc_codes.get(str(merchant_info['mcc']), 'Unknown Category')
                print(f"     카테고리: {category}")
            except:
                print(f"     MCC: {merchant_info['mcc']}")
            
            recommendations.append((merchant_id, rating))
        
        return recommendations
        
    def analyze_latent_factors(self, model_type='svd', n_top_items=5):
        """잠재 요인 분석"""
        print(f"\n=== {model_type.upper()} 잠재 요인 분석 ===")
        
        if model_type == 'svd' and self.svd_model:
            components = self.svd_model.components_
        elif model_type == 'nmf' and self.nmf_model:
            components = self.nmf_model.components_
        else:
            print(f"{model_type} 모델이 학습되지 않았습니다.")
            return
        
        idx_to_item = {idx: item for item, idx in self.item_encoder.items()}
        
        # 각 잠재 요인별로 가장 중요한 상점들 찾기
        for factor_idx in range(min(5, components.shape[0])):  # 상위 5개 요인만
            print(f"\n잠재 요인 {factor_idx + 1}:")
            
            factor_weights = components[factor_idx]
            top_item_indices = np.argsort(factor_weights)[::-1][:n_top_items]
            
            for rank, item_idx in enumerate(top_item_indices, 1):
                merchant_id = idx_to_item[item_idx]
                weight = factor_weights[item_idx]
                
                # 상점 정보
                try:
                    merchant_info = self.transactions_df[self.transactions_df['merchant_id'] == merchant_id].iloc[0]
                    with open(f'{self.data_path}mcc_codes.json', 'r') as f:
                        mcc_codes = json.load(f)
                    category = mcc_codes.get(str(merchant_info['mcc']), 'Unknown')
                    
                    print(f"  {rank}. 상점ID: {merchant_id} (가중치: {weight:.3f})")
                    print(f"      {category} - {merchant_info['merchant_city']}, {merchant_info['merchant_state']}")
                except:
                    print(f"  {rank}. 상점ID: {merchant_id} (가중치: {weight:.3f})")

if __name__ == "__main__":
    # Matrix Factorization 추천 시스템 초기화
    mf = MatrixFactorizationRecommender()
    
    # 1. 데이터 로드 및 준비
    mf.load_and_prepare_data(sample_size=30000)
    
    # 2. 평점 행렬 생성
    mf.create_rating_matrix()
    
    # 3. Train/Test 분할
    mf.train_test_split_matrix()
    
    # 4. SVD 모델 학습
    mf.train_svd_model(n_components=30)
    
    # 5. NMF 모델 학습
    mf.train_nmf_model(n_components=30)
    
    # 6. 모델 평가
    svd_predictions = mf.predict_ratings('svd')
    nmf_predictions = mf.predict_ratings('nmf')
    
    if svd_predictions is not None:
        mf.evaluate_model(svd_predictions, "SVD")
    
    if nmf_predictions is not None:
        mf.evaluate_model(nmf_predictions, "NMF")
    
    # 7. 샘플 사용자에 대한 추천
    sample_users = list(mf.user_encoder.keys())[:3]
    
    for user_id in sample_users:
        # SVD 추천
        mf.get_recommendations(user_id, model_type='svd', n_recommendations=5)
        
        # NMF 추천
        mf.get_recommendations(user_id, model_type='nmf', n_recommendations=5)
        
        print("\n" + "="*80 + "\n")
    
    # 8. 잠재 요인 분석
    mf.analyze_latent_factors('svd')
    mf.analyze_latent_factors('nmf')
    
    print("Day 3 Matrix Factorization 구현이 완료되었습니다!")