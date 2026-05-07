# Clean PINN Advection-Diffusion

Reference implementation for a 1D linear advection-diffusion study:

```text
c_t + v c_x = D c_xx,    x in [0, L], t in [0, T]
```

The code compares:

- Crank-Nicolson reference solver
- Vanilla PINN
- PINN + mass-penalty soft constraint

Recommended experiment roles:

- `dirichlet`: accuracy baseline, not main mass-conservation case.
- `periodic`: clean mass-conservation analysis.
- `zero_flux`: closed-domain conservation, using total flux `J = v c - D c_x`.

## Directory

```text
clean_pinn_advection_diffusion/
  src/advection_diffusion/
    cn_solver.py
    initial_conditions.py
    metrics.py
    pinn_losses.py
    pinn_model.py
    pinn_sampling.py
    plotting.py
  scripts/
    generate_cn_reference.py
    train_pinn.py
    evaluate_model.py
    animate_solution.py
    run_all_cn.py
```

## Example Workflow

Generate CN reference:

```bash
python scripts/generate_cn_reference.py --bc periodic --pe 20 --T 10 --nx 1000 --nt 2000
```

Train Vanilla PINN, usually in Colab/GPU:

```bash
python scripts/train_pinn.py --variant vanilla --bc periodic --pe 20 --T 10 --epochs-adam 10000 --epochs-lbfgs 1000
```

Train Conservative PINN:

```bash
python scripts/train_pinn.py --variant conservative --bc periodic --pe 20 --T 10 --lambda-mass 10
```

Evaluate a trained model against CN reference:

```bash
python scripts/evaluate_model.py ^
  --model outputs/models/pinn_conservative_periodic_Pe20_T10.pt ^
  --reference outputs/reference/reference_periodic_Pe20_T10.npz
```

Make an animation:

```bash
python scripts/animate_solution.py ^
  --reference outputs/reference/reference_periodic_Pe20_T10.npz ^
  --model outputs/models/pinn_conservative_periodic_Pe20_T10.pt
```

## Important Metrics

For conservation analysis, use mass relative to initial mass:

```text
E_cons(t) = |M_method(t) - M0| / |M0|
```

For agreement with reference:

```text
E_ref(t) = |M_PINN(t) - M_CN(t)| / |M0|
```

These answer different questions. Conservation error is the main metric for
periodic/zero-flux cases.
