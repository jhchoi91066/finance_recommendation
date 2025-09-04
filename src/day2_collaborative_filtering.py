import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import json

class CollaborativeFiltering:
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.transactions_df = None
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        
    def load_preprocessed_data(self, sample_size=50000):
        """전처리된 데이터를 로드합니다."""
        print(f"데이터 로딩 중... (샘플 크기: {sample_size})")
        
        # 거래 데이터 로드 (메모리 효율을 위해 샘플링)
        self.transactions_df = pd.read_csv(f'{self.data_path}transactions_data.csv', nrows=sample_size)
        
        # 기본 전처리
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        self.transactions_df['amount'] = self.transactions_df['amount'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # MCC 코드 매핑
        with open(f'{self.data_path}mcc_codes.json', 'r') as f:
            mcc_codes = json.load(f)
        self.transactions_df['mcc_description'] = self.transactions_df['mcc'].astype(str).map(mcc_codes)
        
        print(f"로드된 데이터: {self.transactions_df.shape}")
        print(f"고유 사용자: {self.transactions_df['client_id'].nunique()}")
        print(f"고유 상점: {self.transactions_df['merchant_id'].nunique()}")
        
    def create_user_item_matrix(self):
        """사용자-상품(상점) 상호작용 행렬을 생성합니다."""
        print("\n=== 사용자-아이템 행렬 생성 ===")
        
        # 사용자-상점 거래 빈도 행렬
        user_merchant_counts = self.transactions_df.groupby(['client_id', 'merchant_id']).size().reset_index(name='count')
        
        # 피벗 테이블로 변환
        self.user_item_matrix = user_merchant_counts.pivot(
            index='client_id',
            columns='merchant_id',
            values='count'
        ).fillna(0)
        
        print(f"사용자-아이템 행렬 크기: {self.user_item_matrix.shape}")
        print(f"Sparsity: {(self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100:.2f}%")
        
        # 전치행렬 (아이템-사용자)
        self.item_user_matrix = self.user_item_matrix.T
        
        return self.user_item_matrix
    
    def calculate_user_similarity(self):
        """사용자 간 유사도를 계산합니다."""
        print("\n=== 사용자 유사도 계산 ===")
        
        # 코사인 유사도 계산
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # DataFrame으로 변환
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print(f"사용자 유사도 행렬 크기: {self.user_similarity.shape}")
        
        return self.user_similarity
    
    def calculate_item_similarity(self):
        """아이템(상점) 간 유사도를 계산합니다."""
        print("\n=== 아이템 유사도 계산 ===")
        
        # 코사인 유사도 계산
        self.item_similarity = cosine_similarity(self.item_user_matrix)
        
        # DataFrame으로 변환
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.item_user_matrix.index,
            columns=self.item_user_matrix.index
        )
        
        print(f"아이템 유사도 행렬 크기: {self.item_similarity.shape}")
        
        return self.item_similarity
    
    def user_based_recommendations(self, user_id, n_recommendations=10):
        """사용자 기반 협업 필터링 추천"""
        print(f"\n=== 사용자 {user_id}에 대한 사용자 기반 추천 ===")
        
        if user_id not in self.user_item_matrix.index:
            print(f"사용자 {user_id}가 존재하지 않습니다.")
            return []
        
        # 해당 사용자와 유사한 사용자들 찾기
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:11]  # 자기 제외하고 상위 10명
        
        print(f"가장 유사한 사용자들:")
        for similar_user, similarity_score in similar_users.head(5).items():
            print(f"  사용자 {similar_user}: 유사도 {similarity_score:.4f}")
        
        # 현재 사용자가 이용한 적 없는 상점들 찾기
        user_items = self.user_item_matrix.loc[user_id]
        unvisited_items = user_items[user_items == 0].index
        
        # 유사한 사용자들이 이용한 상점들에 대한 점수 계산
        recommendations = {}
        
        for item in unvisited_items:
            score = 0
            similarity_sum = 0
            
            for similar_user, similarity_score in similar_users.items():
                if self.user_item_matrix.loc[similar_user, item] > 0:
                    score += similarity_score * self.user_item_matrix.loc[similar_user, item]
                    similarity_sum += similarity_score
            
            if similarity_sum > 0:
                recommendations[item] = score / similarity_sum
        
        # 상위 N개 추천
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        print(f"\n추천 상점 (상위 {n_recommendations}개):")
        for i, (merchant_id, score) in enumerate(top_recommendations, 1):
            # 상점 정보 가져오기
            merchant_info = self.transactions_df[self.transactions_df['merchant_id'] == merchant_id].iloc[0]
            print(f"  {i}. 상점ID: {merchant_id}, 점수: {score:.4f}")
            print(f"     위치: {merchant_info['merchant_city']}, {merchant_info['merchant_state']}")
            print(f"     카테고리: {merchant_info['mcc_description']}")
        
        return top_recommendations
    
    def item_based_recommendations(self, user_id, n_recommendations=10):
        """아이템 기반 협업 필터링 추천"""
        print(f"\n=== 사용자 {user_id}에 대한 아이템 기반 추천 ===")
        
        if user_id not in self.user_item_matrix.index:
            print(f"사용자 {user_id}가 존재하지 않습니다.")
            return []
        
        # 사용자가 이용한 상점들
        user_items = self.user_item_matrix.loc[user_id]
        visited_items = user_items[user_items > 0].index.tolist()
        unvisited_items = user_items[user_items == 0].index
        
        print(f"사용자가 이용한 상점 수: {len(visited_items)}")
        
        # 이용한 상점들과 유사한 상점들 찾기
        recommendations = {}
        
        for item in unvisited_items:
            score = 0
            similarity_sum = 0
            
            for visited_item in visited_items:
                similarity_score = self.item_similarity.loc[item, visited_item]
                if similarity_score > 0:
                    score += similarity_score * self.user_item_matrix.loc[user_id, visited_item]
                    similarity_sum += similarity_score
            
            if similarity_sum > 0:
                recommendations[item] = score / similarity_sum
        
        # 상위 N개 추천
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        print(f"\n추천 상점 (상위 {n_recommendations}개):")
        for i, (merchant_id, score) in enumerate(top_recommendations, 1):
            # 상점 정보 가져오기
            merchant_info = self.transactions_df[self.transactions_df['merchant_id'] == merchant_id].iloc[0]
            print(f"  {i}. 상점ID: {merchant_id}, 점수: {score:.4f}")
            print(f"     위치: {merchant_info['merchant_city']}, {merchant_info['merchant_state']}")
            print(f"     카테고리: {merchant_info['mcc_description']}")
        
        return top_recommendations
    
    def get_user_profile(self, user_id):
        """사용자 프로필 분석"""
        print(f"\n=== 사용자 {user_id} 프로필 분석 ===")
        
        if user_id not in self.user_item_matrix.index:
            print(f"사용자 {user_id}가 존재하지 않습니다.")
            return
        
        user_transactions = self.transactions_df[self.transactions_df['client_id'] == user_id]
        
        print(f"총 거래 수: {len(user_transactions)}")
        print(f"총 거래 금액: ${user_transactions['amount'].sum():.2f}")
        print(f"평균 거래 금액: ${user_transactions['amount'].mean():.2f}")
        print(f"이용 상점 수: {user_transactions['merchant_id'].nunique()}")
        
        # 자주 이용하는 카테고리
        top_categories = user_transactions['mcc_description'].value_counts().head(5)
        print("\n자주 이용하는 카테고리:")
        for category, count in top_categories.items():
            print(f"  {category}: {count}회")
        
        # 자주 이용하는 상점
        top_merchants = user_transactions.groupby(['merchant_id', 'merchant_city', 'mcc_description']).size().sort_values(ascending=False).head(5)
        print("\n자주 이용하는 상점:")
        for (merchant_id, city, category), count in top_merchants.items():
            print(f"  상점ID {merchant_id} ({city}): {category}, {count}회")
    
    def evaluate_recommendations(self, test_ratio=0.2):
        """추천 시스템 성능 평가"""
        print(f"\n=== 추천 시스템 성능 평가 ===")
        
        # 간단한 train/test 분할
        users = self.user_item_matrix.index.tolist()
        test_users = np.random.choice(users, size=int(len(users) * test_ratio), replace=False)
        
        precision_scores = []
        recall_scores = []
        
        for user_id in test_users[:10]:  # 처음 10명만 테스트
            # 사용자의 실제 이용 상점들
            actual_items = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
            
            if len(actual_items) < 2:  # 최소 2개 이상의 상점을 이용한 사용자만
                continue
            
            # 절반을 숨기고 나머지로 추천
            hidden_items = set(np.random.choice(list(actual_items), size=max(1, len(actual_items)//2), replace=False))
            remaining_items = actual_items - hidden_items
            
            # 임시로 숨긴 아이템들을 0으로 설정
            temp_matrix = self.user_item_matrix.copy()
            for item in hidden_items:
                temp_matrix.loc[user_id, item] = 0
            
            # 사용자 기반 추천 실행 (간단히 구현)
            try:
                user_recs = self.user_based_recommendations(user_id, n_recommendations=10)
                recommended_items = set([item for item, score in user_recs])
                
                # Precision과 Recall 계산
                hit_items = recommended_items & hidden_items
                precision = len(hit_items) / len(recommended_items) if recommended_items else 0
                recall = len(hit_items) / len(hidden_items) if hidden_items else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                
            except Exception as e:
                continue
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        
        print(f"평균 Precision: {avg_precision:.4f}")
        print(f"평균 Recall: {avg_recall:.4f}")
        print(f"F1-Score: {2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0:.4f}")
        
        return avg_precision, avg_recall

if __name__ == "__main__":
    # 협업 필터링 시스템 초기화
    cf = CollaborativeFiltering()
    
    # 1. 데이터 로드
    cf.load_preprocessed_data(sample_size=30000)  # 메모리 효율을 위해 샘플링
    
    # 2. 사용자-아이템 행렬 생성
    cf.create_user_item_matrix()
    
    # 3. 유사도 계산
    cf.calculate_user_similarity()
    cf.calculate_item_similarity()
    
    # 4. 샘플 사용자에 대한 추천 실행
    sample_users = cf.user_item_matrix.index[:3].tolist()
    
    for user_id in sample_users:
        # 사용자 프로필 분석
        cf.get_user_profile(user_id)
        
        # 사용자 기반 추천
        cf.user_based_recommendations(user_id, n_recommendations=5)
        
        # 아이템 기반 추천
        cf.item_based_recommendations(user_id, n_recommendations=5)
        
        print("\n" + "="*80 + "\n")
    
    # 5. 성능 평가
    # cf.evaluate_recommendations()
    
    print("Day 2 협업 필터링 구현이 완료되었습니다!")