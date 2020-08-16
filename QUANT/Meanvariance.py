# 程序实现基于均值--方差模型的资产配置，包括Markowitz模型和Black-Litterman模型
# 目前均为无卖空约束，后续将陆续增加相关模型，并改进数据读入模块。

from scipy import linalg
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math

stock=pd.read_excel('导出.xlsx')
print(stock.head())


payh = pd.Series(stock.loc[stock.Stkcd=='000001.SZ','Return'].values)
szgt =  pd.Series(stock.loc[stock.Stkcd=='000008.SZ','Return'].values)
dfgx =  pd.Series(stock.loc[stock.Stkcd=='300166.SZ','Return'].values)
jnsw =  pd.Series(stock.loc[stock.Stkcd=='601199.SH','Return'].values)
hqt =  pd.Series(stock.loc[stock.Stkcd=='603116.SH','Return'].values)

# 这里得转化成series才好用axis=1进行拼接
sh_return =pd.concat([payh,szgt,dfgx,jnsw,hqt],axis=1)

# wind导出来单位%，需要手动除以100
sh_return=sh_return/100
sh_return=sh_return.dropna()
cumreturn=(1+sh_return).cumprod()

sh_return.plot()
plt.title('Daily return of 5 stocks)')
plt.legend(loc='lower center', bbox_to_anchor = (0.5,-0.3),
          ncol = 5,fancybox=True, shadow=True)
plt.show()

print("SH RET: LIST")
print(sh_return)
print("CUM RET: LIST")
print(cumreturn.head())

class MeanVariance:
    def __init__(self, returns,rfRet):
        
        self.returns = returns
        self.rfRet=rfRet
        # 计算均值和方差
        self.covs = np.array(self.returns.cov())
        self.means = np.array(self.returns.mean())
    
    # 定义最小化方差的函数，求解二次规划
    # 原理：给定目标收益，求解配置比例w，使得风险最小化。
    # 目标：最小化组合方差
    # 约束：配置比例之和为1；配置比例对应的平均收益率应等于目标收益率goal return。
    # 该问题可以等价转化为如此优化函数，具体参考：https://www.jianshu.com/p/ae6bb3ea0d41
    # 返回配置的比例

    def minVar(self, goalRet):

        # append是矩阵拼接，如果是0就上下拼接，1就是左右拼接；类似concat
        L1 = np.append(np.append(self.covs.swapaxes(0,1),[self.means],0),
                      [np.ones(len(self.means))],0).swapaxes(0,1)
        L2 = list(np.ones(len(self.means)))
        L2.extend([0,0])
        L3 = list(self.means)
        L3.extend([0,0])
        L4 = np.array([L2,L3])
        L = np.append(L1,L4,0)
        results = linalg.solve(L,np.append(np.zeros(len(self.means)),[1,goalRet],0))
        return (np.array([list(self.returns.columns), results[:-2]]))
    

    # 定义绘制最小方差曲线函数
    # 同时计算Sharpe ratio，寻找切线组合，在图中标出并返回最大夏普比率、切点组合目标收益率和标准差
    def frontierCurve(self):
        goals = [x/500000 for x in range(-3500, 4000)]
        stdvar = list(map(lambda x: self.calVar(self.minVar(x)[1,:].astype(np.float)),goals))
        plt.plot(stdvar, goals)
        # 寻找切点组合
        SharpeRatio=list(map(lambda x,y: (x-self.rfRet)/y,goals,stdvar))
        Max_Sharpe=max(SharpeRatio)
        Max_Sharpe_index=SharpeRatio.index(Max_Sharpe)
        # 标注切点组合
        plt.scatter(stdvar[Max_Sharpe_index],goals[Max_Sharpe_index])
        plt.show()
        return [Max_Sharpe,goals[Max_Sharpe_index],stdvar[Max_Sharpe_index]]
    
    def blacklitterman(self,tau,P,Q):
        # 基本原理请参考链接：https://zhuanlan.zhihu.com/p/38282835
        covs = self.covs
        means = self.means
        pil = np.expand_dims(means,axis = 0).T
        ts = tau * covs
        ts_1 = linalg.inv(ts)
        Omega = np.dot(np.dot(P,ts), P.T)* np.eye(Q.shape[0])
        Omega_1 = linalg.inv(Omega)
        er = np.dot(linalg.inv(ts_1 + np.dot(np.dot(P.T,Omega_1),P)),(np.dot(ts_1 ,pil)+np.dot(np.dot(P.T,Omega_1),Q)))
        posterirorSigma = linalg.inv(ts_1 + np.dot(np.dot(P.T,Omega_1),P))
        return [er, posterirorSigma]

    def blminVar(self,blres, goalRet):
        covs = self.covs
        means = self.means
        L1 = np.append(np.append(covs.swapaxes(0,1),[means.flatten()],axis=0),
                    [np.ones(len(means))],axis=0).swapaxes(0,1)

        L2 = list(np.ones(len(means)))
        L2.extend([0,0])
        L3 = list(means)
        L3.extend([0,0])
        L4 = np.array([L2,L3],dtype=float)
        L = np.append(L1,L4,axis=0)
        results = linalg.solve(L,np.append(np.zeros(len(means)),[1,goalRet]))

        return pd.DataFrame(results[:-2],columns = ['p_weight'])
    #给定个资产的比例，计算收益率均值

    def meanRet(self, fracs):
        meanRisky = self.returns.mean()
        assert len(meanRisky) == len(fracs), 'Length of fractions must be equal to nuber of assets'
        return(np.sum(np.multiply(meanRisky, np.array(fracs))))
    
    #给定各资产的比例，计算收益率方差
    def calVar(self, fracs):
        return (math.sqrt(np.dot(np.dot(fracs, self.returns.cov()), fracs)))

minVar = MeanVariance(sh_return,0.025/365)

# c=minVar.frontierCurve()
# print(c)

pick1 = np.array([1,0,1,1,1])
q1 = np.array([0.003*4])
pick2 = np.array([0.5,0.5,0,0,-1])
q2 = np.array([0.001])
P = np.array([pick1,pick2])
Q = np.array([q1,q2])

c=minVar.blacklitterman(0.1,P,Q)
blresult=minVar.blminVar(c,0.2/252)
print(blresult)