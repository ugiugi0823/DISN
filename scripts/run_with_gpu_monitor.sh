#!/bin/bash

max_memory=0

# GPU 0번의 메모리 사용량을 모니터링하는 함수
monitor_gpu() {
    while true; do
        memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
        echo "현재 GPU 0 (NVIDIA A100 80GB PCIe) 메모리 사용량: ${memory}MiB"
        if [[ $memory =~ ^[0-9]+$ ]]; then
            if (( memory > max_memory )); then
                max_memory=$memory
                echo "새로운 최대 메모리 사용량: ${max_memory}MiB"
            fi
        else
            echo "경고: 유효하지 않은 메모리 값: $memory"
        fi
        sleep 1
    done
}

# 백그라운드에서 GPU 모니터링 시작
monitor_gpu &
monitor_pid=$!

echo "GPU 모니터링 시작. PID: $monitor_pid"

# GPU 정보 출력
echo "사용 중인 GPU 정보:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 원래의 Python 스크립트 실행
echo "Python 스크립트 실행 시작"
CUDA_VISIBLE_DEVICES=0 python run_dataset.py \
--original_dataset_path "./original_dataset" \
--new_dataset_path "./new_dataset" \
--prompt "photo of a crack defect image" \
--neg_prompt " " \
--datacheck \
--bigger
echo "Python 스크립트 실행 완료"

# Python 스크립트 종료 후 모니터링 중지
echo "GPU 모니터링 중지"
kill $monitor_pid

# 잠시 대기하여 백그라운드 프로세스가 완전히 종료되도록 함
sleep 2

echo "GPU 0 (NVIDIA A100 80GB PCIe)의 최종 최대 메모리 사용량: ${max_memory}MiB"
echo "총 가용 메모리: 81920MiB"
echo "최대 메모리 사용률: $((max_memory * 100 / 81920))%"