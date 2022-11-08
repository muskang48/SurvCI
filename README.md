# Learn to Live Longer: Counterfactual Inference using Balanced Representations for Parametric Deep Survival Analysis

### Dataset Description
Following are the datasets used in the paper:

**1. Synthetic** - For generating synthetic dataset, we employed the synthetic dataset discussed in [SurvITE](https://github.com/SamuelHakansson/survITE).The ground truth time-to-event and time-to-censoring outcomes were generated
using Restricted Mean Survival Time (RMST) as follows:
```math
y^a_{T,i} \textstyle{=} \textstyle{\int}^{T^{*}}_0 {S}^a(y^a_{T,i}|\mathbf{x})dy_{T,i}
      \quad \textrm{and} \quad 
      y^a_{C,i} \textstyle{=} \textstyle{\int}^{T^{*}}_0 {S}^a(y^a_{C,i}|\mathbf{x})dy_{C,i}
```
where individual specific and censoring survival functions are computed as  
```math 
S^a(y|\mathbf{x}_i) = \displaystyle{\prod}_{k \leq t}(1-h^a(y|\mathbf{x}_i))
``` 
for both  $y_{T,i}$  and $y_{C,i}$ for all $i$. Towards computing $\epsilon_{ATE}$ and $\epsilon_{PEHE}$, the ground truth ITE is given by $\tau(\mathbf{x}\_i;\tilde{L}) = \min(y^1_{T,i},\tilde{L}) - \min(y^0_{T,i},\tilde{L})$ and the estimated ITE is given as 
```math 
\hat{\tau}(\mathbf{x};\tilde{L}) = \textstyle{\int}_{y^a_{T,i} = 0}^{\tilde{L}} \hat{S}^1(y^a_{T,i}|\mathbf{x}) - \hat{S}^0(y^a_{T,i}|\mathbf{x}) dy^a_{T,i},
```
i.e., ITE is computed over time epochs at which events have been reported up to $\tilde{L}$, as specified in the paper.

**2. ACTG-Semi Synthetic** - For generating ACTG Semi Synthetic Dataset, we used the ACTG discussed in [CSA](https://github.com/paidamoyo/counterfactual_survival_analysis). The time-to-event is generated as $y^a_{T,i} \sim \frac{1}{\alpha^a}\log(1-\tfrac{\alpha^a\log U}{\lambda^a\exp(\mathbf{x}^T\beta_a))})$. To simulate informative censoring, time-to-censoring is generated as $\log(Y_{C,i}) \sim \tfrac{1}{\alpha^a_C}\log(1-\tfrac{\alpha^a_c\log U}{\lambda_c^a\exp(\mathbf{x}^T\beta_a))}$, where $U \sim Unif(0,1)$, $\alpha_c^a$ and $\alpha_a= 5e^{-3}$, $\lambda_a = 6e^{-4}$ and $\lambda_c^a = 8.8e^{-4}$. Further, we assign the instance as censored $\delta = 0$ if $y^a_{T,i} > y^a_{C,i}$ and uncensored if $y^a_{T,i} < y^a_{C,i}$. In the case of ACTG, we do not compute ground-truth survival function and the time-to-censoring and time-to-event is computed directly using equations.

*We divide the synthetic dataset with 50% of the data reserved for training and the rest for testing. Further 30% of training data is taken as validation set. For ACTG, we split the data into training, validation and test sets according to 70%, 15% , 15% partitions respectively.*

### Hyperparameter Tuning 
To train the SurvCI model we performed hyper parameter tuning and used Adam optimizer in all the experiments. In all experiments we set the Scaling ELBO Censored Loss $\alpha = 1$, i.e., we give equal importance to event and censored data. In spite of this setting, we see from simulations that the effect of bias in larger quantiles is not too high. We use the Linear MMD as the balancing IPM Term. Although the choice between Log-Normal or Weibull distribution can be treated as a  hyper parameter, we have considered Log-Normal for all experiments. The representation learning function $\Phi(.)$ is a fully connected Multi- Layer Perceptron with dimension $[100,100]$. The number of mixture distribution components, $K$, is chosen from $[3,6]$. All experiments were conducted in PyTorch.

Hyperparameters used in experiments pertaining SurvCI Model for different datasets

| Datasets | K | Scaling IPM | Scaling SE | Scaling ELBO|  Scaling L2 |Batch Size| Learning Rate |
| ---------|---:|----:| -----|----:| -----:|-----:|----:|
| Synthetic,S1 | 3 | 0.001 | 0.1 | 1|0.5 | 200 | 3e-4|
| Synthetic,S2 | 3 | 0.001 | 0.1 | 1|0.5 |200 | 3e-4 |
| Synthetic,s3 | 3 | 0.5 | 2e-4 | 1|0.2 |100| 3e-4 |
| Synthetic,S4 | 3 | 0.5 | 3e-4 | 1|0.2 | 100 | 3e-4|
| ACTG,S3 | 3 | 0.5 | 2e-4 |1| 0.2 |1497 | 3e-4 |
| ACTG,s4 | 3 | 0.5 | 2e-4 |1| 0.2 |1497| 3e-4 |

Hyperparameters used in experiments pertaining SurvCI-Info Model for different datasets

| Datasets | K | Scaling IPM | Scaling SE| Scaling ELBO | Scaling L2 |Batch Size| Learning Rate |
| ---------|---:|----:| -----|----:| -----:|-----:|----:|
| Synthetic,S2 | 3 | 10 | 3e-5 | 0.6| 0.2 |200 | 3e-4 |
| Synthetic,S4 | 3 | 10 | 3e-5 | 0.6| 0.2 | 100 | 3e-4|
| ACTG,s4 | 6 | 1 | 1e-6 | 0.5|0.2 |64| 3e-5 |

```
**NOTE** For running experiments for S1 and S2 setting we force few samples as treated samples in each batch during training to avoid Runtime error
```
