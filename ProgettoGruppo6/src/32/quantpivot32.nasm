default rel

section .text
global approx_distance_asm
global euclidean_distance_asm
global compute_lower_bound_asm

; ============================================================
; approx_distance_asm - VERSIONE PRODUCTION READY
;   RDI = vplus
;   RSI = vminus
;   RDX = wplus
;   RCX = wminus
;   R8D = D
;
; OTTIMIZZAZIONI:
;   - Unrolling x4 (16 float/iterazione)
;   - MOVUPS per sicurezza (funziona anche con dati non allineati)
;   - Prefetching aggressivo
;   - Gestione robusta remainder (16 -> 8 -> 4 -> 1)
;
; Ritorno in XMM0 (float)
; ============================================================

approx_distance_asm:
    ; Reset accumulatori
    xorps xmm0, xmm0        ; sum_pp
    xorps xmm1, xmm1        ; sum_mm
    xorps xmm2, xmm2        ; sum_pm
    xorps xmm3, xmm3        ; sum_mp

    ; Test edge case
    test r8d, r8d
    jz .return_zero

    ; ----------------------------------------
    ; Loop principale: 16 float/iterazione
    ; ----------------------------------------
    mov eax, r8d
    shr eax, 4              ; eax = D/16
    jz .check8

.main_loop16:
    ; Prefetch aggressivo
    prefetchnta [rdi + 256]
    prefetchnta [rsi + 256]
    prefetchnta [rdx + 256]
    prefetchnta [rcx + 256]

    ; ==== BLOCCO 1 (0-3) - MOVUPS per sicurezza ====
    movups xmm4, [rdi]      ; SAFE: funziona anche se non allineato
    movups xmm5, [rsi]
    movups xmm6, [rdx]
    movups xmm7, [rcx]

    movaps xmm8, xmm4       ; register-to-register (veloce)
    mulps  xmm8, xmm6
    addps  xmm0, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm7
    addps  xmm1, xmm8

    movaps xmm8, xmm4
    mulps  xmm8, xmm7
    addps  xmm2, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm6
    addps  xmm3, xmm8

    ; ==== BLOCCO 2 (4-7) ====
    movups xmm4, [rdi + 16]
    movups xmm5, [rsi + 16]
    movups xmm6, [rdx + 16]
    movups xmm7, [rcx + 16]

    movaps xmm8, xmm4
    mulps  xmm8, xmm6
    addps  xmm0, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm7
    addps  xmm1, xmm8

    movaps xmm8, xmm4
    mulps  xmm8, xmm7
    addps  xmm2, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm6
    addps  xmm3, xmm8

    ; ==== BLOCCO 3 (8-11) ====
    movups xmm4, [rdi + 32]
    movups xmm5, [rsi + 32]
    movups xmm6, [rdx + 32]
    movups xmm7, [rcx + 32]

    movaps xmm8, xmm4
    mulps  xmm8, xmm6
    addps  xmm0, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm7
    addps  xmm1, xmm8

    movaps xmm8, xmm4
    mulps  xmm8, xmm7
    addps  xmm2, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm6
    addps  xmm3, xmm8

    ; ==== BLOCCO 4 (12-15) ====
    movups xmm4, [rdi + 48]
    movups xmm5, [rsi + 48]
    movups xmm6, [rdx + 48]
    movups xmm7, [rcx + 48]

    movaps xmm8, xmm4
    mulps  xmm8, xmm6
    addps  xmm0, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm7
    addps  xmm1, xmm8

    movaps xmm8, xmm4
    mulps  xmm8, xmm7
    addps  xmm2, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm6
    addps  xmm3, xmm8

    ; Avanza 16 float = 64 byte
    add rdi, 64
    add rsi, 64
    add rdx, 64
    add rcx, 64

    dec eax
    jnz .main_loop16

; ----------------------------------------
; Remainder: 8 float
; ----------------------------------------
.check8:
    mov eax, r8d
    shr eax, 3
    and eax, 1
    jz .check4

.main_loop8:
    movups xmm4, [rdi]
    movups xmm5, [rsi]
    movups xmm6, [rdx]
    movups xmm7, [rcx]

    movaps xmm8, xmm4
    mulps  xmm8, xmm6
    addps  xmm0, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm7
    addps  xmm1, xmm8

    movaps xmm8, xmm4
    mulps  xmm8, xmm7
    addps  xmm2, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm6
    addps  xmm3, xmm8

    movups xmm4, [rdi + 16]
    movups xmm5, [rsi + 16]
    movups xmm6, [rdx + 16]
    movups xmm7, [rcx + 16]

    movaps xmm8, xmm4
    mulps  xmm8, xmm6
    addps  xmm0, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm7
    addps  xmm1, xmm8

    movaps xmm8, xmm4
    mulps  xmm8, xmm7
    addps  xmm2, xmm8

    movaps xmm8, xmm5
    mulps  xmm8, xmm6
    addps  xmm3, xmm8

    add rdi, 32
    add rsi, 32
    add rdx, 32
    add rcx, 32

; ----------------------------------------
; Remainder: 4 float
; ----------------------------------------
.check4:
    mov eax, r8d
    shr eax, 2
    and eax, 1
    jz .check1

.main_loop4:
    movups xmm4, [rdi]
    movups xmm5, [rsi]
    movups xmm6, [rdx]
    movups xmm7, [rcx]

    movaps xmm8, xmm4
    mulps xmm8, xmm6
    addps xmm0, xmm8

    movaps xmm8, xmm5
    mulps xmm8, xmm7
    addps xmm1, xmm8

    movaps xmm8, xmm4
    mulps xmm8, xmm7
    addps xmm2, xmm8

    movaps xmm8, xmm5
    mulps xmm8, xmm6
    addps xmm3, xmm8

    add rdi, 16
    add rsi, 16
    add rdx, 16
    add rcx, 16

; ----------------------------------------
; Remainder: 1-3 float
; ----------------------------------------
.check1:
    mov eax, r8d
    and eax, 3
    jz .reduce_all

.remainder_loop:
    movss xmm4, [rdi]
    movss xmm5, [rsi]
    movss xmm6, [rdx]
    movss xmm7, [rcx]

    movaps xmm8, xmm4
    mulss xmm8, xmm6
    addss xmm0, xmm8

    movaps xmm8, xmm5
    mulss xmm8, xmm7
    addss xmm1, xmm8

    movaps xmm8, xmm4
    mulss xmm8, xmm7
    addss xmm2, xmm8

    movaps xmm8, xmm5
    mulss xmm8, xmm6
    addss xmm3, xmm8

    add rdi, 4
    add rsi, 4
    add rdx, 4
    add rcx, 4

    dec eax
    jnz .remainder_loop

; ============================================================
; RIDUZIONE ORIZZONTALE
; ============================================================
.reduce_all:
    ; sum_pp -> xmm0
    movaps xmm8, xmm0
    shufps xmm8, xmm8, 0x4E
    addps  xmm0, xmm8
    movaps xmm8, xmm0
    shufps xmm8, xmm8, 0xB1
    addps  xmm0, xmm8

    ; sum_mm -> xmm1
    movaps xmm8, xmm1
    shufps xmm8, xmm8, 0x4E
    addps  xmm1, xmm8
    movaps xmm8, xmm1
    shufps xmm8, xmm8, 0xB1
    addps  xmm1, xmm8

    ; sum_pm -> xmm2
    movaps xmm8, xmm2
    shufps xmm8, xmm8, 0x4E
    addps  xmm2, xmm8
    movaps xmm8, xmm2
    shufps xmm8, xmm8, 0xB1
    addps  xmm2, xmm8

    ; sum_mp -> xmm3
    movaps xmm8, xmm3
    shufps xmm8, xmm8, 0x4E
    addps  xmm3, xmm8
    movaps xmm8, xmm3
    shufps xmm8, xmm8, 0xB1
    addps  xmm3, xmm8

    ; Risultato finale
    addss xmm0, xmm1
    addss xmm2, xmm3
    subss xmm0, xmm2

.return_zero:
    ret


; ============================================================
; euclidean_distance_asm - VERSIONE PRODUCTION READY
;   RDI = v
;   RSI = w
;   EDX = D
;
; OTTIMIZZAZIONI:
;   - Unrolling x4 (16 float/iterazione)
;   - MOVUPS per sicurezza
;   - Prefetching
;   - Sqrt finale ottimizzata
; ============================================================

euclidean_distance_asm:
    xorps xmm0, xmm0        ; sum_sq = 0

    ; Test edge case
    test edx, edx
    jz .return_zero

    ; ----------------------------------------
    ; Loop principale: 16 float/iterazione
    ; ----------------------------------------
    mov eax, edx
    shr eax, 4              ; eax = D/16
    jz .check8

.main_loop16:
    prefetchnta [rdi + 256]
    prefetchnta [rsi + 256]

    ; Blocco 1 - MOVUPS per sicurezza
    movups xmm1, [rdi]
    movups xmm2, [rsi]
    subps  xmm1, xmm2
    mulps  xmm1, xmm1
    addps  xmm0, xmm1

    ; Blocco 2
    movups xmm1, [rdi + 16]
    movups xmm2, [rsi + 16]
    subps  xmm1, xmm2
    mulps  xmm1, xmm1
    addps  xmm0, xmm1

    ; Blocco 3
    movups xmm1, [rdi + 32]
    movups xmm2, [rsi + 32]
    subps  xmm1, xmm2
    mulps  xmm1, xmm1
    addps  xmm0, xmm1

    ; Blocco 4
    movups xmm1, [rdi + 48]
    movups xmm2, [rsi + 48]
    subps  xmm1, xmm2
    mulps  xmm1, xmm1
    addps  xmm0, xmm1

    add rdi, 64
    add rsi, 64

    dec eax
    jnz .main_loop16

; ----------------------------------------
; Remainder: 8 float
; ----------------------------------------
.check8:
    mov eax, edx
    shr eax, 3
    and eax, 1
    jz .check4

.main_loop8:
    movups xmm1, [rdi]
    movups xmm2, [rsi]
    subps  xmm1, xmm2
    mulps  xmm1, xmm1
    addps  xmm0, xmm1

    movups xmm1, [rdi + 16]
    movups xmm2, [rsi + 16]
    subps  xmm1, xmm2
    mulps  xmm1, xmm1
    addps  xmm0, xmm1

    add rdi, 32
    add rsi, 32

; ----------------------------------------
; Remainder: 4 float
; ----------------------------------------
.check4:
    mov eax, edx
    shr eax, 2
    and eax, 1
    jz .check1

.main_loop4:
    movups xmm1, [rdi]
    movups xmm2, [rsi]
    subps  xmm1, xmm2
    mulps  xmm1, xmm1
    addps  xmm0, xmm1

    add rdi, 16
    add rsi, 16

; ----------------------------------------
; Remainder: 1-3 float
; ----------------------------------------
.check1:
    mov eax, edx
    and eax, 3
    jz .reduce

.remainder_loop:
    movss xmm1, [rdi]
    movss xmm2, [rsi]
    subss xmm1, xmm2
    mulss xmm1, xmm1
    addss xmm0, xmm1

    add rdi, 4
    add rsi, 4

    dec eax
    jnz .remainder_loop

; ============================================================
; RIDUZIONE E SQRT
; ============================================================
.reduce:
    ; Riduzione orizzontale
    movaps xmm1, xmm0
    shufps xmm1, xmm1, 0x4E
    addps  xmm0, xmm1
    
    movaps xmm1, xmm0
    shufps xmm1, xmm1, 0xB1
    addps  xmm0, xmm1

    ; Sqrt finale
    sqrtss xmm0, xmm0

.return_zero:
    ret


; ============================================================
; compute_lower_bound_asm - VERSIONE PRODUCTION READY
;   RDI = idx_v (array di h float)
;   RSI = qpivot (array di h float)
;   EDX = h
;
; Calcola: LB = max_j |idx_v[j] - qpivot[j]|
; ============================================================

compute_lower_bound_asm:
    ; Test edge case
    test edx, edx
    jz .return_zero

    ; Maschera per valore assoluto
    mov rax, 0x7FFFFFFF
    movd xmm7, eax
    shufps xmm7, xmm7, 0    ; [mask, mask, mask, mask]

    xorps xmm0, xmm0        ; max_LB = 0

    ; ----------------------------------------
    ; Loop principale: 4 float alla volta
    ; ----------------------------------------
    mov eax, edx
    shr eax, 2              ; eax = h/4
    jz .check1

.main_loop4:
    movups xmm1, [rdi]      ; SAFE: movups
    movups xmm2, [rsi]
    subps  xmm1, xmm2       ; diff
    andps  xmm1, xmm7       ; |diff|
    maxps  xmm0, xmm1       ; max element-wise

    add rdi, 16
    add rsi, 16

    dec eax
    jnz .main_loop4

; ----------------------------------------
; Remainder: 1-3 elementi
; ----------------------------------------
.check1:
    mov eax, edx
    and eax, 3
    jz .reduce

.remainder_loop:
    movss xmm1, [rdi]
    movss xmm2, [rsi]
    subss xmm1, xmm2
    andps xmm1, xmm7        ; abs
    maxss xmm0, xmm1

    add rdi, 4
    add rsi, 4

    dec eax
    jnz .remainder_loop

; ============================================================
; RIDUZIONE ORIZZONTALE (trova max)
; ============================================================
.reduce:
    movaps xmm1, xmm0
    shufps xmm1, xmm1, 0x4E
    maxps  xmm0, xmm1
    
    movaps xmm1, xmm0
    shufps xmm1, xmm1, 0xB1
    maxss  xmm0, xmm1

.return_zero:
    ret