---
license: apache-2.0
language:
- en
tags:
- in-context-reinforcement-learning
- reinforcement-learning
- in-context-learning
- metaworld
- mujoco
- bi-dexhands
- industrial-benchmark
model-index:
  - name: dunnolab/Vintix
    results:
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: MuJoCo
          type: mujoco
        metrics:
          - type: iqm_normalized_95
            value: 0.99
            name: Normalized Score IQM (95% CI)
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: Meta-World
          type: metaworld
        metrics:
          - type: iqm_normalized_95
            value: 0.99
            name: Normalized Score IQM (95% CI)
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: Bi-DexHands
          type: bi-dexhands
        metrics:
          - type: iqm_normalized_95
            value: 0.92
            name: Normalized Score IQM (95% CI)
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: Industrial-Benchmark
          type: industrial-benchmark
        metrics:
          - type: iqm_normalized_95
            value: 0.99
            name: Normalized Score IQM (95% CI)
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: ant_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 6315.00 +/- 675.00
            name: Total reward
          - type: normalized_total_reward
            value: 0.98 +/- 0.10
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: halfcheetah_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 7226.50 +/- 241.50
            name: Total reward
          - type: normalized_total_reward
            value: 0.93 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: hopper_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 2794.60 +/- 612.62
            name: Total reward
          - type: normalized_total_reward
            value: 0.86 +/- 0.19
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: humanoid_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 7376.26 +/- 0.00
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.00
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: humanoidstandup_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 320567.82 +/- 58462.11
            name: Total reward
          - type: normalized_total_reward
            value: 1.02 +/- 0.21
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: inverteddoublependulum_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 6105.75 +/- 4368.65
            name: Total reward
          - type: normalized_total_reward
            value: 0.65 +/- 0.47
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: invertedpendulum_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 1000.00 +/- 0.00
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.00
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: pusher_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: -37.82 +/- 8.72
            name: Total reward
          - type: normalized_total_reward
            value: 1.02 +/- 0.08
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: reacher_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: -6.25 +/- 2.63
            name: Total reward
          - type: normalized_total_reward
            value: 0.98 +/- 0.07
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: swimmer_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 93.20 +/- 5.40
            name: Total reward
          - type: normalized_total_reward
            value: 0.98 +/- 0.06
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: walker2d_v4
          type: MuJoCo
        metrics:
          - type: total_reward
            value: 5400.00 +/- 107.95
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: assembly-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 307.08 +/- 25.20
            name: Total reward
          - type: normalized_total_reward
            value: 1.04 +/- 0.10
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: basketball-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 568.04 +/- 60.72
            name: Total reward
          - type: normalized_total_reward
            value: 1.02 +/- 0.11
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: bin-picking-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 7.88 +/- 4.28
            name: Total reward
          - type: normalized_total_reward
            value: 0.01 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: box-close-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 61.75 +/- 13.54
            name: Total reward
          - type: normalized_total_reward
            value: -0.04 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: button-press-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 624.67 +/- 42.77
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.07
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: button-press-topdown-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 449.36 +/- 62.16
            name: Total reward
          - type: normalized_total_reward
            value: 0.94 +/- 0.14
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: button-press-topdown-wall-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 482.08 +/- 32.48
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.07
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: button-press-wall-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 672.00 +/- 26.48
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.04
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: coffee-button-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 719.00 +/- 41.10
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.06
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: coffee-pull-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 26.04 +/- 56.12
            name: Total reward
          - type: normalized_total_reward
            value: 0.07 +/- 0.20
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: coffee-push-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 571.01 +/- 112.28
            name: Total reward
          - type: normalized_total_reward
            value: 1.01 +/- 0.20
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: dial-turn-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 783.90 +/- 53.17
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.07
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: disassemble-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 523.60 +/- 58.15
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.12
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: door-close-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 538.10 +/- 25.76
            name: Total reward
          - type: normalized_total_reward
            value: 1.02 +/- 0.05
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: door-lock-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 356.51 +/- 249.44
            name: Total reward
          - type: normalized_total_reward
            value: 0.35 +/- 0.36
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: door-open-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 581.33 +/- 26.33
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.05
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: door-unlock-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 352.86 +/- 147.78
            name: Total reward
          - type: normalized_total_reward
            value: 0.21 +/- 0.26
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: drawer-close-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 838.88 +/- 7.41
            name: Total reward
          - type: normalized_total_reward
            value: 0.96 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: drawer-open-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 493.00 +/- 3.57
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: faucet-close-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 749.46 +/- 14.83
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: faucet-open-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 732.47 +/- 15.23
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: hammer-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 669.31 +/- 69.56
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.12
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: hand-insert-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 142.81 +/- 146.64
            name: Total reward
          - type: normalized_total_reward
            value: 0.19 +/- 0.20
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: handle-press-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 835.30 +/- 114.19
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.15
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: handle-press-side-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 852.96 +/- 16.08
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: handle-pull-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 701.10 +/- 13.82
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: handle-pull-side-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 493.10 +/- 53.65
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.11
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: lever-pull-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 548.72 +/- 81.12
            name: Total reward
          - type: normalized_total_reward
            value: 0.96 +/- 0.16
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: peg-insert-side-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 352.43 +/- 137.24
            name: Total reward
          - type: normalized_total_reward
            value: 1.01 +/- 0.40
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: peg-unplug-side-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 401.52 +/- 175.27
            name: Total reward
          - type: normalized_total_reward
            value: 0.75 +/- 0.34
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: pick-out-of-hole-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 364.20 +/- 79.56
            name: Total reward
          - type: normalized_total_reward
            value: 0.91 +/- 0.20
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: pick-place-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 414.02 +/- 91.10
            name: Total reward
          - type: normalized_total_reward
            value: 0.98 +/- 0.22
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: pick-place-wall-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 553.18 +/- 84.72
            name: Total reward
          - type: normalized_total_reward
            value: 1.04 +/- 0.16
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: plate-slide-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 531.98 +/- 156.94
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.34
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: plate-slide-back-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 703.93 +/- 108.27
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.16
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: plate-slide-back-side-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 721.29 +/- 62.15
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.09
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: plate-slide-side-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 578.24 +/- 143.73
            name: Total reward
          - type: normalized_total_reward
            value: 0.83 +/- 0.22
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: push-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 729.33 +/- 104.40
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.14
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: push-back-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 372.16 +/- 112.75
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.29
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: push-wall-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 741.68 +/- 14.84
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: reach-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 684.45 +/- 136.55
            name: Total reward
          - type: normalized_total_reward
            value: 1.01 +/- 0.26
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: reach-wall-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 738.02 +/- 100.96
            name: Total reward
          - type: normalized_total_reward
            value: 0.98 +/- 0.17
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shelf-place-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 268.34 +/- 29.07
            name: Total reward
          - type: normalized_total_reward
            value: 1.01 +/- 0.11
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: soccer-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 438.44 +/- 189.63
            name: Total reward
          - type: normalized_total_reward
            value: 0.80 +/- 0.35
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: stick-pull-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 483.98 +/- 83.25
            name: Total reward
          - type: normalized_total_reward
            value: 0.92 +/- 0.16
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: stick-push-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 563.07 +/- 173.40
            name: Total reward
          - type: normalized_total_reward
            value: 0.90 +/- 0.28
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: sweep-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 487.19 +/- 60.02
            name: Total reward
          - type: normalized_total_reward
            value: 0.94 +/- 0.12
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: sweep-into-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 798.80 +/- 15.62
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: window-close-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 562.48 +/- 91.17
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.17
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: window-open-v2
          type: Meta-World
        metrics:
          - type: total_reward
            value: 573.69 +/- 93.98
            name: Total reward
          - type: normalized_total_reward
            value: 0.96 +/- 0.17
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandblockstack
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 347.40 +/- 50.60
            name: Total reward
          - type: normalized_total_reward
            value: 1.17 +/- 0.23
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandbottlecap
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 338.25 +/- 81.25
            name: Total reward
          - type: normalized_total_reward
            value: 0.81 +/- 0.25
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandcatchabreast
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 11.81 +/- 21.28
            name: Total reward
          - type: normalized_total_reward
            value: 0.17 +/- 0.32
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandcatchover2underarm
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 31.60 +/- 7.20
            name: Total reward
          - type: normalized_total_reward
            value: 0.92 +/- 0.24
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandcatchunderarm
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 18.21 +/- 9.46
            name: Total reward
          - type: normalized_total_reward
            value: 0.72 +/- 0.39
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhanddoorcloseinward
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 3.97 +/- 0.15
            name: Total reward
          - type: normalized_total_reward
            value: 0.36 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhanddoorcloseoutward
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 358.50 +/- 4.50
            name: Total reward
          - type: normalized_total_reward
            value: -1.27 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhanddooropeninward
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 108.25 +/- 8.50
            name: Total reward
          - type: normalized_total_reward
            value: 0.29 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhanddooropenoutward
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 83.65 +/- 12.10
            name: Total reward
          - type: normalized_total_reward
            value: 0.13 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandgraspandplace
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 485.15 +/- 89.10
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.18
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandkettle
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: -450.47 +/- 0.00
            name: Total reward
          - type: normalized_total_reward
            value: -0.99 +/- 0.00
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandliftunderarm
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 377.92 +/- 13.24
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandover
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 33.01 +/- 0.96
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandpen
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 98.80 +/- 83.60
            name: Total reward
          - type: normalized_total_reward
            value: 0.52 +/- 0.44
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandpushblock
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 445.60 +/- 2.20
            name: Total reward
          - type: normalized_total_reward
            value: 0.98 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandreorientation
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 2798.00 +/- 2112.00
            name: Total reward
          - type: normalized_total_reward
            value: 0.89 +/- 0.66
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandscissors
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 747.95 +/- 7.65
            name: Total reward
          - type: normalized_total_reward
            value: 1.03 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandswingcup
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 3775.50 +/- 583.70
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.13
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandswitch
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 268.25 +/- 2.35
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: shadowhandtwocatchunderarm
          type: Bi-DexHands
        metrics:
          - type: total_reward
            value: 2.17 +/- 0.67
            name: Total reward
          - type: normalized_total_reward
            value: 0.03 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-0-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -191.39 +/- 22.96
            name: Total reward
          - type: normalized_total_reward
            value: 0.94 +/- 0.13
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-5-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -194.01 +/- 3.66
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-10-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -213.28 +/- 2.01
            name: Total reward
          - type: normalized_total_reward
            value: 1.01 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-15-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -227.82 +/- 4.29
            name: Total reward
          - type: normalized_total_reward
            value: 1.01 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-20-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -259.99 +/- 22.70
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.11
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-25-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -282.28 +/- 20.70
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.11
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-30-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -307.02 +/- 19.23
            name: Total reward
          - type: normalized_total_reward
            value: 0.90 +/- 0.10
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-35-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -314.36 +/- 5.62
            name: Total reward
          - type: normalized_total_reward
            value: 1.00 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-40-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -339.34 +/- 9.57
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.05
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-45-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -366.63 +/- 7.47
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.04
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-50-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -395.94 +/- 17.65
            name: Total reward
          - type: normalized_total_reward
            value: 0.91 +/- 0.09
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-55-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -403.73 +/- 2.03
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-60-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -434.25 +/- 4.12
            name: Total reward
          - type: normalized_total_reward
            value: 0.98 +/- 0.02
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-65-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -480.31 +/- 8.63
            name: Total reward
          - type: normalized_total_reward
            value: 0.86 +/- 0.04
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-70-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -480.76 +/- 5.98
            name: Total reward
          - type: normalized_total_reward
            value: 0.95 +/- 0.03
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-75-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -476.83 +/- 2.44
            name: Total reward
          - type: normalized_total_reward
            value: 0.99 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-80-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -497.13 +/- 2.95
            name: Total reward
          - type: normalized_total_reward
            value: 0.96 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-85-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -513.83 +/- 3.06
            name: Total reward
          - type: normalized_total_reward
            value: 0.98 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-90-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -532.70 +/- 3.61
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-95-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -557.42 +/- 3.81
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.01
            name: Expert normalized total reward
      - task:
          type: in-context-reinforcement-learning
          name: In-Context Reinforcement Learning
        dataset:
          name: industrial-benchmark-100-v1
          type: Industrial-Benchmark
        metrics:
          - type: total_reward
            value: -574.57 +/- 4.37
            name: Total reward
          - type: normalized_total_reward
            value: 0.97 +/- 0.01
            name: Expert normalized total reward
---
# Model Card for Vintix

This is a multi-task action model via in-context reinforcement learning

## Model Details
| Setting                         | Description                                                    |
|---------------------------------|----------------------------------------------------------------|
| Parameters                      | 332M                                                           |
| Model Sizes                     | Layers: 20, Heads: 16, Embedding Size: 1024                    |
| Sequence Length                 | 8192                                                           |
| Training Data                   | MuJoCo, Meta-World, Bi-DexHands, Industrial Benchmark          |

### Model Description
- Developed by: [dunnolab](https://dunnolab.ai)
- License: Apache 2.0

### Model Sources
- Repository: https://github.com/dunnolab/vintix
- Paper: https://arxiv.org/abs/2501.19400

## Citation

```bibex
@article{polubarov2025vintix,
  author={Andrey Polubarov and Nikita Lyubaykin and Alexander Derevyagin and Ilya Zisman and Denis Tarasov and Alexander Nikulin and Vladislav Kurenkov},
  title={Vintix: Action Model via In-Context Reinforcement Learning},
  journal={arXiv},
  volume={2501.19400},
  year={2025}
}
```