import numpy as np

def calculate_bank_conflicts(bm, bn, bk, tm, tn):
    # 파라미터 설정
    num_banks = 32
    warp_size = 32
    
    # 스레드 ID 생성
    num_threads = (bm // tm) * (bn // tn)
    tid = np.arange(num_threads)
    
    # trow와 tcol 계산
    trow = tid // (bn // tn)
    tcol = tid % (bn // tn)
    
    def analyze_conflicts(memory_accesses, access_name):
        conflicts = {}
        for warp_start in range(0, len(tid), warp_size):
            warp_end = min(warp_start + warp_size, len(tid))
            warp_accesses = memory_accesses[warp_start:warp_end]
            bank_accesses = warp_accesses % num_banks
            unique, counts = np.unique(bank_accesses, return_counts=True)
            max_conflict = counts.max()
            if max_conflict > 1:
                conflicts[warp_start // warp_size] = max_conflict
        
        if conflicts:
            print(f"{access_name} Bank Conflicts:")
            for warp, conflict in conflicts.items():
                print(f"  Warp {warp}: {conflict}-way conflict")
        else:
            print(f"{access_name}: No bank conflicts detected")

    # A_shared 접근 분석
    for j in range(bk):
        A_accesses = ((trow * tm + np.arange(tm)[:, np.newaxis]) * bk + j).flatten()
        analyze_conflicts(A_accesses, "A_shared")

    # B_shared 접근 분석
    for j in range(bk):
        B_accesses = (j * bn + tcol * tn + np.arange(tn)[:, np.newaxis]).flatten()
        analyze_conflicts(B_accesses, "B_shared")

# 파라미터 설정
bm, bn, bk = 256, 128, 16
tm, tn = 8, 8

calculate_bank_conflicts(bm, bn, bk, tm, tn)