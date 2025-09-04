#!/usr/bin/env python3
"""
Finance Recommendation System - Main Entry Point

이 스크립트는 전체 추천 시스템 파이프라인을 실행합니다.
사용자는 이 파일을 통해 전체 프로세스를 한 번에 실행할 수 있습니다.

Usage:
    python src/main.py --step [1|2|3|4|all] --sample_size [int]
    
    --step: 실행할 단계 (1: 전처리, 2: 협업필터링, 3: 행렬분해, 4: 평가, all: 전체)
    --sample_size: 샘플 데이터 크기 (기본값: 30000)
"""

import argparse
import sys
import time
from pathlib import Path

# 현재 디렉토리를 PATH에 추가
sys.path.append(str(Path(__file__).parent))

def run_data_preprocessing(sample_size=30000):
    """데이터 전처리 실행"""
    print("=" * 60)
    print("Step 1: 데이터 전처리 실행")
    print("=" * 60)
    
    try:
        from day1_quick_test import *
        print("✅ 데이터 전처리 완료")
        return True
    except Exception as e:
        print(f"❌ 데이터 전처리 실패: {e}")
        return False

def run_collaborative_filtering(sample_size=30000):
    """협업 필터링 실행"""
    print("=" * 60)
    print("Step 2: 협업 필터링 실행")
    print("=" * 60)
    
    try:
        from day2_collaborative_filtering import CollaborativeFiltering
        
        cf = CollaborativeFiltering()
        cf.load_preprocessed_data(sample_size=min(sample_size, 20000))
        cf.create_user_item_matrix()
        cf.calculate_user_similarity()
        cf.calculate_item_similarity()
        
        # 샘플 추천 실행
        sample_users = list(cf.user_item_matrix.index)[:2]
        for user_id in sample_users:
            cf.get_user_profile(user_id)
            cf.user_based_recommendations(user_id, n_recommendations=3)
            cf.item_based_recommendations(user_id, n_recommendations=3)
            
        print("✅ 협업 필터링 완료")
        return True
    except Exception as e:
        print(f"❌ 협업 필터링 실패: {e}")
        return False

def run_matrix_factorization(sample_size=30000):
    """행렬 분해 실행"""
    print("=" * 60)
    print("Step 3: 행렬 분해 실행")
    print("=" * 60)
    
    try:
        from day3_matrix_factorization import MatrixFactorizationRecommender
        
        mf = MatrixFactorizationRecommender()
        mf.load_and_prepare_data(sample_size=min(sample_size, 20000))
        mf.create_rating_matrix()
        mf.train_test_split_matrix()
        mf.train_svd_model(n_components=20)
        mf.train_nmf_model(n_components=20)
        
        # 성능 평가
        svd_predictions = mf.predict_ratings('svd')
        nmf_predictions = mf.predict_ratings('nmf')
        
        if svd_predictions is not None:
            mf.evaluate_model(svd_predictions, "SVD")
        if nmf_predictions is not None:
            mf.evaluate_model(nmf_predictions, "NMF")
            
        print("✅ 행렬 분해 완료")
        return True
    except Exception as e:
        print(f"❌ 행렬 분해 실패: {e}")
        return False

def run_evaluation(sample_size=30000):
    """모델 평가 실행"""
    print("=" * 60)
    print("Step 4: 모델 평가 실행")
    print("=" * 60)
    
    try:
        from day4_evaluation import RecommendationEvaluator
        
        evaluator = RecommendationEvaluator()
        evaluator.load_data(sample_size=min(sample_size, 25000))
        
        # 모델별 평가
        evaluator.evaluate_collaborative_filtering(sample_size=min(sample_size, 15000))
        evaluator.evaluate_matrix_factorization(sample_size=min(sample_size, 15000))
        
        # 결과 비교 및 저장
        evaluator.compare_models()
        evaluator.generate_insights()
        evaluator.save_results()
        
        print("✅ 모델 평가 완료")
        return True
    except Exception as e:
        print(f"❌ 모델 평가 실패: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Finance Recommendation System')
    parser.add_argument('--step', choices=['1', '2', '3', '4', 'all'], default='all',
                        help='실행할 단계 선택')
    parser.add_argument('--sample_size', type=int, default=30000,
                        help='샘플 데이터 크기')
    
    args = parser.parse_args()
    
    print("🚀 Finance Recommendation System 시작")
    print(f"📊 샘플 크기: {args.sample_size:,}")
    print(f"📋 실행 단계: {args.step}")
    print()
    
    start_time = time.time()
    success_count = 0
    total_steps = 0
    
    steps = {
        '1': ('데이터 전처리', run_data_preprocessing),
        '2': ('협업 필터링', run_collaborative_filtering), 
        '3': ('행렬 분해', run_matrix_factorization),
        '4': ('모델 평가', run_evaluation)
    }
    
    # 실행할 단계 결정
    if args.step == 'all':
        run_steps = list(steps.keys())
    else:
        run_steps = [args.step]
    
    # 단계별 실행
    for step in run_steps:
        total_steps += 1
        step_name, step_func = steps[step]
        
        print(f"\n🔄 {step_name} 시작...")
        step_start = time.time()
        
        if step_func(args.sample_size):
            success_count += 1
            step_time = time.time() - step_start
            print(f"✅ {step_name} 완료 (소요시간: {step_time:.1f}초)")
        else:
            step_time = time.time() - step_start
            print(f"❌ {step_name} 실패 (소요시간: {step_time:.1f}초)")
    
    # 최종 결과
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🏁 실행 완료!")
    print("=" * 60)
    print(f"✅ 성공: {success_count}/{total_steps} 단계")
    print(f"⏱️  총 소요시간: {total_time:.1f}초")
    
    if success_count == total_steps:
        print("🎉 모든 단계가 성공적으로 완료되었습니다!")
        print("\n📁 결과 파일:")
        print("   - ./reports/evaluation_results.json")
        print("   - ./reports/model_comparison.csv")
        print("\n📖 자세한 내용은 README.md를 확인하세요.")
    else:
        print("⚠️  일부 단계에서 오류가 발생했습니다.")
        print("    로그를 확인하고 필요한 경우 개별 스크립트를 실행하세요.")
    
    return success_count == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)