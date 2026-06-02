# RecSys
RecSys Challenge 2026

# 1. inference 실행
python run_inference_devset.py --tid llama1b_hybrid_devset --batch_size 16
# → exp/inference/devset/llama1b_hybrid_devset_MMDD_HHMM.json

# 2. ground truth 없으면 먼저 생성 (처음 한 번만)
python make_ground_truth.py
# → exp/ground_truth/devset.json

# 3. 평가
python evaluate_devset.py --tid llama1b_hybrid_devset_MMDD_HHMM --eval_dataset devset
# → exp/scores/devset/llama1b_hybrid_devset_MMDD_HHMM.json

# 4. 결과 확인
python analyze_results.py --pred exp/inference/devset/llama1b_hybrid_devset_MMDD_HHMM.json