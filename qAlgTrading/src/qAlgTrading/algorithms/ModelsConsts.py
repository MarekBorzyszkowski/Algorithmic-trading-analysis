from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from qiskit_machine_learning.algorithms import QSVR, QSVC

LINEAR_REG = 'LinearRegression'
SVC_ALG = 'SVC'
QSVC_ALG = 'QSVC'
SVR_ALG = 'SVR'
QSVR_ALG = 'QSVR'

MODELS = {
    LINEAR_REG: LinearRegression(),
    SVC_ALG: SVC(),
    QSVC_ALG: QSVC(),
    SVR_ALG: SVR(),
    QSVR_ALG: QSVR()
}
CLASSIFIERS = [SVC_ALG, QSVC_ALG]
REGRESSORS = [LINEAR_REG, SVR_ALG, QSVR_ALG]
SVR_REGRESSORS = [SVR_ALG, QSVR_ALG]
SVM_ALGORITHMS = [SVC_ALG, QSVC_ALG, SVR_ALG, QSVR_ALG]