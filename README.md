# load_prediction

## xgboost

XGBoost全称是eXtreme Gradient Boosting，是一种梯度增强形式的集成学习，与bagging集成学习的方式不同，bagging的方式是通过对样本或特征进行选择抽样，从而组合成不同的训练集来进行训练，可以得到若干存在差异的模型，最后在预测时会将输入送给每个模型，将各个模型的输出进行组合汇总，得到一个更为准确的预测结果。Gradient Boosting的方式虽然同样会产生若干个模型，但模型需要按照一定的顺序进行预测，除了第一个模型拟合训练集的标签外，后面每个模型的目标是拟合上一个模型的残差，最终将所有模型的输出加在一起就可以得到一个更加准确的预测结果。

大体流程如下：

    1. 真实标签：y
    2. 模型1预测目标：y，预测结果：y'
    3. 模型2预测目标：y-y' 预测结果：y''
    4. 模型3预测目标：y-y'-y'' 预测结果：y'''
    5. 最终模型预测输出：y'+y''+y'''≈y'+y''+y-y'-y''≈y

在模型的选择上，通常会选择决策树(Decision Tree)作为基模型，因为决策树通过超参数的配置更容易产生差异较大的模型，从而增强模型的鲁棒性。

Gradient Boosting是一种机器学习算法，而XGBoost是Gradient Boosting的一种工程化实现。相比传统的GBDT(Gradient Boosting Decision Tree)，XGBoost在以下几个方面做出了改善：

    1. 以树模型作为基准模型时，XGBoost显式地加入了正则项来控制模型的复杂度，有利于防止过拟合，从而提高模型的泛化能力。
    2. GBDT在模型训练时只使用了损失函数的一阶导数信息，XGBoost对损失函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。
    3. 传统的GBDT通常采用决策树这种CART（Classification And Regression Tree）作为基分类|回归器，XGBoost支持多种类型的基模型，比如线性分类|回归器。
    4. 传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机森林相似的策略，支持对数据进行采样。
    5. 传统GBDT没有设计对缺失值进行处理，XGBoost能够通过稀疏感知算法自动学习出缺失值的处理策略。
    6. 传统GBDT没有进行并行化设计，注意不是tree维度的并行，而是特征维度的并行。XGBoost预先将每个特征按特征值排好序，存储为块结构，分裂结点时可以采用多线程并行查找每个特征的最佳分割点，极大提升训练速度。

## 贝叶斯超参数优化
所谓超参数是用来控制学习过程的不同参数值对机器学习模型的性能影响，可简单理解为是预先设定好的一组模型参数。如在XGBoost中，超参数有n_estimators（基准模型的数量）、max_depth（每颗树的最大深度）等等，这些是在训练之前预先设定好的。超参数优化就是寻找合适的超参数值组合，使得能够最大化模型的准确率和鲁棒性。它对机器学习算法的预测精度起着至关重要的作用。传统常见的超参数优化策略有网格搜索或随机搜索，网格搜索是针对人为设定的参数空间，遍历所有的参数组合，选择给定指标（如MSE、R2_SCORE等）最值点对应的那组参数。当参数空间比较大时，这种方式会比较耗时。与网格搜索相比，随机搜索并未尝试所有参数值，而是从指定的分布中采样固定数量的参数设置。它的理论依据是，如果随机样本点集足够大，那么也可以找到全局的最大或最小值，或它们的近似值。通过对搜索范围的随机取样，随机搜索一般会比网格搜索要快一些，但无法保证可以找到全局的最值点。

贝叶斯优化作为一种高级的超参数优化算法，在速度和模型准确性上都带来了比较好的效果。贝叶斯优化假设超参数与最后我们需要优化的损失函数存在一个函数关系，并假设该函数关系满足高斯分布。其基本原理是通过采样一定数量的参数组合计算后验概率，来估计下一个参数的选择方向，选择后不断的更新参数选择范围的后验概率，通过类似启发式算法的方式有针对性的选择下一组可能让模型预测效果更好的参数。这种方式相比随机采样搜索，能够有效的保证参数选择的合理性，同时大大减少参数选择所需的搜索时间。

## MCMC采样
该采样方法结合了马尔可夫链和蒙特卡罗随机的思想，作为机器学习模型的辅助算法。假设每个小时的冷量服从不同的分布，从历史每小时冷量占比分布中进行采样，得到一个符合历史分布的值。详细公式推导参考可参考该链接[MCMC采样算法原理](https://www.cnblogs.com/pinard/p/6625739.html)