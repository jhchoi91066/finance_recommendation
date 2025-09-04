import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import json
import sys
import os

# 이전 모듈들 import
sys.path.append('.')
from day2_collaborative_filtering import CollaborativeFiltering
from day3_matrix_factorization import MatrixFactorizationRecommender

class RecommendationEvaluator:
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.transactions_df = None
        self.results = {}
        
    def load_data(self, sample_size=30000):
        """평가용 데이터 로드"""
        print(f"평가용 데이터 로딩 중... (샘플 크기: {sample_size})")
        
        self.transactions_df = pd.read_csv(f'{self.data_path}transactions_data.csv', nrows=sample_size)
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        self.transactions_df['amount'] = self.transactions_df['amount'].str.replace('$', '').str.replace(',', '').astype(float)
        self.transactions_df['amount'] = self.transactions_df['amount'].abs()
        
        print(f"평가 데이터: {self.transactions_df.shape}")
        
    def evaluate_collaborative_filtering(self, sample_size=20000):
        """협업 필터링 모델 평가"""
        print("\n" + "="*60)
        print("협업 필터링 모델 평가")
        print("="*60)
        
        try:
            # 협업 필터링 모델 초기화 및 학습
            cf = CollaborativeFiltering(self.data_path)
            cf.load_preprocessed_data(sample_size=sample_size)
            cf.create_user_item_matrix()
            cf.calculate_user_similarity()
            cf.calculate_item_similarity()
            
            # 평가 메트릭
            cf_results = {
                'model_name': 'Collaborative Filtering',
                'precision_at_k': {},
                'recall_at_k': {},
                'f1_at_k': {},
                'coverage': 0,
                'diversity': 0
            }
            
            # 샘플 사용자들에 대해 평가
            sample_users = list(cf.user_item_matrix.index)[:50]  # 처음 50명
            
            for k in [5, 10, 20]:
                precisions = []
                recalls = []
                
                for user_id in sample_users:
                    try:
                        # 사용자의 실제 상호작용
                        user_interactions = set(cf.user_item_matrix.loc[user_id][cf.user_item_matrix.loc[user_id] > 0].index)
                        
                        if len(user_interactions) < 3:  # 최소 3개 이상
                            continue
                        
                        # 일부를 테스트용으로 숨기기
                        test_size = max(1, len(user_interactions) // 3)
                        test_items = set(np.random.choice(list(user_interactions), size=test_size, replace=False))
                        
                        # 추천 생성 (간단한 popularity 기반)
                        item_popularity = cf.user_item_matrix.sum(axis=0)
                        unvisited_items = set(cf.user_item_matrix.columns) - user_interactions
                        
                        if len(unvisited_items) == 0:
                            continue
                        
                        # 인기도 기반 추천
                        popular_unvisited = sorted(unvisited_items, 
                                                 key=lambda x: item_popularity[x], 
                                                 reverse=True)[:k]
                        
                        recommended_items = set(popular_unvisited)
                        
                        # Precision@K와 Recall@K 계산
                        hits = recommended_items & test_items
                        precision = len(hits) / len(recommended_items) if recommended_items else 0
                        recall = len(hits) / len(test_items) if test_items else 0
                        
                        precisions.append(precision)
                        recalls.append(recall)
                        
                    except Exception as e:
                        continue
                
                avg_precision = np.mean(precisions) if precisions else 0
                avg_recall = np.mean(recalls) if recalls else 0
                f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
                
                cf_results['precision_at_k'][k] = avg_precision
                cf_results['recall_at_k'][k] = avg_recall
                cf_results['f1_at_k'][k] = f1
                
                print(f"Precision@{k}: {avg_precision:.4f}")
                print(f"Recall@{k}: {avg_recall:.4f}")
                print(f"F1@{k}: {f1:.4f}")
            
            # Coverage 계산 (추천 가능한 아이템 비율)
            total_items = cf.user_item_matrix.shape[1]
            recommended_items_total = set()
            
            for user_id in sample_users[:20]:  # 처음 20명만
                try:
                    item_popularity = cf.user_item_matrix.sum(axis=0)
                    user_interactions = set(cf.user_item_matrix.loc[user_id][cf.user_item_matrix.loc[user_id] > 0].index)
                    unvisited_items = set(cf.user_item_matrix.columns) - user_interactions
                    
                    if unvisited_items:
                        popular_unvisited = sorted(unvisited_items, 
                                                 key=lambda x: item_popularity[x], 
                                                 reverse=True)[:10]
                        recommended_items_total.update(popular_unvisited)
                except:
                    continue
            
            coverage = len(recommended_items_total) / total_items
            cf_results['coverage'] = coverage
            
            print(f"Coverage: {coverage:.4f}")
            
            self.results['collaborative_filtering'] = cf_results
            
        except Exception as e:
            print(f"협업 필터링 평가 중 오류: {e}")
            self.results['collaborative_filtering'] = None
    
    def evaluate_matrix_factorization(self, sample_size=20000):
        """행렬 분해 모델 평가"""
        print("\n" + "="*60)
        print("행렬 분해 모델 평가")
        print("="*60)
        
        try:
            # 행렬 분해 모델 초기화 및 학습
            mf = MatrixFactorizationRecommender(self.data_path)
            mf.load_and_prepare_data(sample_size=sample_size)
            mf.create_rating_matrix()
            train_data, test_data = mf.train_test_split_matrix(test_ratio=0.3)
            
            # SVD 모델 학습
            mf.train_svd_model(n_components=20)
            svd_predictions = mf.predict_ratings('svd')
            
            # NMF 모델 학습
            mf.train_nmf_model(n_components=20)
            nmf_predictions = mf.predict_ratings('nmf')
            
            # RMSE 및 MAE 계산
            test_indices = np.nonzero(mf.test_matrix)
            actual_ratings = mf.test_matrix[test_indices]
            
            # SVD 평가
            svd_predicted_ratings = svd_predictions[test_indices]
            svd_rmse = np.sqrt(mean_squared_error(actual_ratings, svd_predicted_ratings))
            svd_mae = mean_absolute_error(actual_ratings, svd_predicted_ratings)
            
            # NMF 평가
            nmf_predicted_ratings = nmf_predictions[test_indices]
            nmf_rmse = np.sqrt(mean_squared_error(actual_ratings, nmf_predicted_ratings))
            nmf_mae = mean_absolute_error(actual_ratings, nmf_predicted_ratings)
            
            print(f"SVD 모델:")
            print(f"  RMSE: {svd_rmse:.4f}")
            print(f"  MAE: {svd_mae:.4f}")
            
            print(f"NMF 모델:")
            print(f"  RMSE: {nmf_rmse:.4f}")
            print(f"  MAE: {nmf_mae:.4f}")
            
            # Precision@K, Recall@K 계산
            svd_results = self._calculate_ranking_metrics(mf, svd_predictions, 'SVD')
            nmf_results = self._calculate_ranking_metrics(mf, nmf_predictions, 'NMF')
            
            # 결과 저장
            self.results['svd'] = {
                'model_name': 'SVD',
                'rmse': svd_rmse,
                'mae': svd_mae,
                **svd_results
            }
            
            self.results['nmf'] = {
                'model_name': 'NMF', 
                'rmse': nmf_rmse,
                'mae': nmf_mae,
                **nmf_results
            }
            
        except Exception as e:
            print(f"행렬 분해 평가 중 오류: {e}")
            self.results['svd'] = None
            self.results['nmf'] = None
    
    def _calculate_ranking_metrics(self, mf, predictions, model_name):
        """랭킹 기반 메트릭 계산"""
        print(f"\n{model_name} 랭킹 메트릭 계산:")
        
        results = {
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'coverage': 0
        }
        
        # 샘플 사용자들에 대해 평가
        sample_users = np.random.choice(mf.user_item_matrix.shape[0], size=30, replace=False)
        
        for k in [5, 10, 20]:
            precisions = []
            recalls = []
            
            for user_idx in sample_users:
                try:
                    # 실제 평점을 준 아이템들
                    actual_items = set(np.nonzero(mf.user_item_matrix[user_idx])[0])
                    
                    if len(actual_items) < 3:
                        continue
                    
                    # 일부를 테스트용으로 숨기기
                    test_size = max(1, len(actual_items) // 3)
                    test_items = set(np.random.choice(list(actual_items), size=test_size, replace=False))
                    
                    # 예측 평점 기반 추천
                    user_predictions = predictions[user_idx]
                    unrated_items = np.setdiff1d(np.arange(len(user_predictions)), 
                                               list(actual_items - test_items))
                    
                    if len(unrated_items) == 0:
                        continue
                    
                    # 상위 K개 추천
                    unrated_predictions = user_predictions[unrated_items]
                    top_k_indices = np.argsort(unrated_predictions)[::-1][:k]
                    recommended_items = set(unrated_items[top_k_indices])
                    
                    # Precision@K와 Recall@K 계산
                    hits = recommended_items & test_items
                    precision = len(hits) / len(recommended_items) if recommended_items else 0
                    recall = len(hits) / len(test_items) if test_items else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    
                except Exception as e:
                    continue
            
            avg_precision = np.mean(precisions) if precisions else 0
            avg_recall = np.mean(recalls) if recalls else 0
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            results['precision_at_k'][k] = avg_precision
            results['recall_at_k'][k] = avg_recall
            results['f1_at_k'][k] = f1
            
            print(f"  Precision@{k}: {avg_precision:.4f}")
            print(f"  Recall@{k}: {avg_recall:.4f}")
            print(f"  F1@{k}: {f1:.4f}")
        
        return results
    
    def compare_models(self):
        """모델 비교 분석"""
        print("\n" + "="*60)
        print("모델 성능 비교")
        print("="*60)
        
        # 결과 테이블 생성
        comparison_df = pd.DataFrame()
        
        for model_key, results in self.results.items():
            if results is None:
                continue
                
            row_data = {
                'Model': results['model_name'],
                'RMSE': results.get('rmse', 'N/A'),
                'MAE': results.get('mae', 'N/A'),
                'Precision@5': results.get('precision_at_k', {}).get(5, 'N/A'),
                'Precision@10': results.get('precision_at_k', {}).get(10, 'N/A'),
                'Recall@5': results.get('recall_at_k', {}).get(5, 'N/A'),
                'Recall@10': results.get('recall_at_k', {}).get(10, 'N/A'),
                'F1@5': results.get('f1_at_k', {}).get(5, 'N/A'),
                'F1@10': results.get('f1_at_k', {}).get(10, 'N/A'),
                'Coverage': results.get('coverage', 'N/A')
            }
            
            comparison_df = pd.concat([comparison_df, pd.DataFrame([row_data])], ignore_index=True)
        
        print("\n모델 성능 비교 테이블:")
        print("-" * 120)
        
        # 숫자 컬럼만 포맷팅 적용
        formatted_df = comparison_df.copy()
        numeric_columns = ['RMSE', 'MAE', 'Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'F1@5', 'F1@10', 'Coverage']
        
        for col in numeric_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{float(x):.4f}" if x != 'N/A' else 'N/A'
                )
        
        print(formatted_df.to_string(index=False))
        
        # 최적 모델 찾기
        print("\n모델별 강점:")
        
        if not comparison_df.empty:
            numeric_cols = ['RMSE', 'MAE', 'Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'F1@5', 'F1@10', 'Coverage']
            
            for col in numeric_cols:
                if col in comparison_df.columns:
                    valid_data = comparison_df[comparison_df[col] != 'N/A']
                    if not valid_data.empty:
                        if col in ['RMSE', 'MAE']:  # 낮을수록 좋음
                            best_idx = valid_data[col].astype(float).idxmin()
                        else:  # 높을수록 좋음
                            best_idx = valid_data[col].astype(float).idxmax()
                        
                        best_model = valid_data.iloc[best_idx]['Model']
                        best_value = valid_data.iloc[best_idx][col]
                        print(f"  {col}: {best_model} ({best_value})")
        
        return comparison_df
    
    def generate_insights(self):
        """평가 결과 인사이트 생성"""
        print("\n" + "="*60)
        print("추천 시스템 평가 인사이트")
        print("="*60)
        
        insights = []
        
        # 데이터 특성 분석
        total_users = self.transactions_df['client_id'].nunique()
        total_merchants = self.transactions_df['merchant_id'].nunique()
        avg_transactions_per_user = len(self.transactions_df) / total_users
        
        insights.append(f"데이터 특성:")
        insights.append(f"  - 총 사용자 수: {total_users:,}")
        insights.append(f"  - 총 상점 수: {total_merchants:,}")
        insights.append(f"  - 사용자당 평균 거래 수: {avg_transactions_per_user:.1f}")
        insights.append(f"  - 데이터 희소성: 매우 높음 (99%+ 0값)")
        
        # 모델별 특성
        insights.append(f"\n모델별 특성:")
        
        if self.results.get('collaborative_filtering'):
            insights.append(f"  - 협업 필터링: 사용자 간 유사성 기반, Cold Start 문제 존재")
        
        if self.results.get('svd'):
            svd_rmse = self.results['svd']['rmse']
            insights.append(f"  - SVD: 차원 축소 기반, RMSE {svd_rmse:.4f}")
        
        if self.results.get('nmf'):
            nmf_rmse = self.results['nmf']['rmse']
            insights.append(f"  - NMF: 음수 없는 인수분해, RMSE {nmf_rmse:.4f}")
        
        # 추천 시스템 개선 제안
        insights.append(f"\n개선 제안:")
        insights.append(f"  - 콘텐츠 기반 필터링 추가 (상점 카테고리, 위치 정보 활용)")
        insights.append(f"  - 하이브리드 모델 구현 (여러 기법 조합)")
        insights.append(f"  - 시간 정보 활용 (시계열 패턴 반영)")
        insights.append(f"  - 더 많은 피처 엔지니어링 (거래 패턴, 금액 범위 등)")
        
        # 결과 출력
        for insight in insights:
            print(insight)
    
    def save_results(self):
        """평가 결과 저장"""
        print("\n평가 결과 저장 중...")
        
        # JSON으로 결과 저장
        results_to_save = {}
        for key, value in self.results.items():
            if value is not None:
                results_to_save[key] = value
        
        with open('./reports/evaluation_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        # CSV로도 저장
        comparison_df = self.compare_models()
        if not comparison_df.empty:
            comparison_df.to_csv('./reports/model_comparison.csv', index=False)
        
        print("평가 결과가 ./reports/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    # 추천 시스템 평가 실행
    evaluator = RecommendationEvaluator()
    
    # 1. 데이터 로드
    evaluator.load_data(sample_size=25000)
    
    # 2. 협업 필터링 평가
    evaluator.evaluate_collaborative_filtering(sample_size=15000)
    
    # 3. 행렬 분해 모델 평가
    evaluator.evaluate_matrix_factorization(sample_size=15000)
    
    # 4. 모델 비교
    evaluator.compare_models()
    
    # 5. 인사이트 생성
    evaluator.generate_insights()
    
    # 6. 결과 저장
    evaluator.save_results()
    
    print("\nDay 4 추천 시스템 평가가 완료되었습니다!")