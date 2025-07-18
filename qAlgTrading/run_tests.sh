#!/bin/bash
export QISKIT_PARALLEL=True
export QISKIT_NUM_PROCS=30
export QISKIT_FORCE_THREADS=True


if [ ! -d venv ]; then
  echo "venv hasn't been initialized. Running setup.sh"
  ./setup.sh
fi
source venv/bin/activate


#python3 src/train.py jsons/wig_20.json pca LinearRegression true
#python3 src/train.py jsons/wig_20.json pca SVC true
#python3 src/train.py jsons/wig_20.json pca SVR true
#python3 src/train.py jsons/wig_20.json qpca LinearRegression true
#python3 src/train.py jsons/wig_20.json qpca SVC true
#python3 src/train.py jsons/wig_20.json qpca SVR true

#python3 src/signalsFromRegresion.py jsons/S\&P.json svr none none
#python3 src/signalsFromRegresion.py jsons/wig_20.json svr none none
#python3 src/signalsFromRegresion.py jsons/S\&P.json qsvr none none
#python3 src/signalsFromRegresion.py jsons/wig_20.json qsvr none none
#python3 src/signalsFromRegresion.py jsons/S\&P.json lr none none
#python3 src/signalsFromRegresion.py jsons/wig_20.json lr none none
#python3 src/signalsFromRegresion.py jsons/S\&P.json pca LinearRegression true
#python3 src/signalsFromRegresion.py jsons/wig_20.json pca LinearRegression true
#python3 src/signalsFromRegresion.py jsons/S\&P.json pca SVR true
#python3 src/signalsFromRegresion.py jsons/wig_20.json pca SVR true
#python3 src/signalsFromRegresion.py jsons/S\&P.json pca QSVR true
#python3 src/signalsFromRegresion.py jsons/wig_20.json pca QSVR true
#python3 src/signalsFromRegresion.py jsons/S\&P.json qpca LinearRegression true
#python3 src/signalsFromRegresion.py jsons/wig_20.json qpca LinearRegression true
#python3 src/signalsFromRegresion.py jsons/S\&P.json qpca SVR true
#python3 src/signalsFromRegresion.py jsons/wig_20.json qpca SVR true
#python3 src/signalsFromRegresion.py jsons/S\&P.json qpca QSVR true
#python3 src/signalsFromRegresion.py jsons/wig_20.json qpca QSVR true


#python3 src/statisticsForSignals.py jsons/S\&P.json lr none none
#python3 src/statisticsForSignals.py jsons/wig_20.json lr none none
#python3 src/statisticsForSignals.py jsons/S\&P.json svr none none
#python3 src/statisticsForSignals.py jsons/wig_20.json svr none none
#python3 src/statisticsForSignals.py jsons/S\&P.json svc none none
#python3 src/statisticsForSignals.py jsons/wig_20.json svc none none
#python3 src/statisticsForSignals.py jsons/S\&P.json qsvr none none
#python3 src/statisticsForSignals.py jsons/wig_20.json qsvr none none
#python3 src/statisticsForSignals.py jsons/S\&P.json qsvc none none
#python3 src/statisticsForSignals.py jsons/wig_20.json qsvc none none
#python3 src/statisticsForSignals.py jsons/S\&P.json pca LinearRegression true
#python3 src/statisticsForSignals.py jsons/wig_20.json pca LinearRegression true
#python3 src/statisticsForSignals.py jsons/S\&P.json pca SVR true
#python3 src/statisticsForSignals.py jsons/wig_20.json pca SVR true
#python3 src/statisticsForSignals.py jsons/S\&P.json pca QSVR true
#python3 src/statisticsForSignals.py jsons/wig_20.json pca QSVR true
#python3 src/statisticsForSignals.py jsons/S\&P.json pca SVC true
#python3 src/statisticsForSignals.py jsons/wig_20.json pca SVC true
#python3 src/statisticsForSignals.py jsons/S\&P.json pca QSVC true
#python3 src/statisticsForSignals.py jsons/wig_20.json pca QSVC true
#python3 src/statisticsForSignals.py jsons/S\&P.json qpca LinearRegression true
#python3 src/statisticsForSignals.py jsons/wig_20.json qpca LinearRegression true
#python3 src/statisticsForSignals.py jsons/S\&P.json qpca SVR true
#python3 src/statisticsForSignals.py jsons/wig_20.json qpca SVR true
#python3 src/statisticsForSignals.py jsons/S\&P.json qpca QSVR true
#python3 src/statisticsForSignals.py jsons/wig_20.json qpca QSVR true
#python3 src/statisticsForSignals.py jsons/S\&P.json qpca SVC true
#python3 src/statisticsForSignals.py jsons/wig_20.json qpca SVC true
#python3 src/statisticsForSignals.py jsons/S\&P.json qpca QSVC true
#python3 src/statisticsForSignals.py jsons/wig_20.json qpca QSVC true


python3 src/trade.py jsons/S\&P.json perfect none none &
python3 src/trade.py jsons/wig_20.json perfect none none &
python3 src/trade.py jsons/S\&P.json macd none none &
python3 src/trade.py jsons/wig_20.json macd none none &
python3 src/trade.py jsons/S\&P.json bb none none &
python3 src/trade.py jsons/wig_20.json bb none none &
python3 src/trade.py jsons/S\&P.json lr none none &
python3 src/trade.py jsons/wig_20.json lr none none &
python3 src/trade.py jsons/S\&P.json svr none none &
python3 src/trade.py jsons/wig_20.json svr none none &
python3 src/trade.py jsons/S\&P.json svc none none &
python3 src/trade.py jsons/wig_20.json svc none none &
python3 src/trade.py jsons/S\&P.json qsvr none none &
python3 src/trade.py jsons/wig_20.json qsvr none none &
python3 src/trade.py jsons/S\&P.json qsvc none none &
python3 src/trade.py jsons/wig_20.json qsvc none none &
python3 src/trade.py jsons/S\&P.json pca LinearRegression true &
python3 src/trade.py jsons/wig_20.json pca LinearRegression true &
python3 src/trade.py jsons/S\&P.json pca SVR true &
python3 src/trade.py jsons/wig_20.json pca SVR true &
python3 src/trade.py jsons/S\&P.json pca QSVR true &
python3 src/trade.py jsons/wig_20.json pca QSVR true &
python3 src/trade.py jsons/S\&P.json pca SVC true &
python3 src/trade.py jsons/wig_20.json pca SVC true &
python3 src/trade.py jsons/S\&P.json pca QSVC true &
python3 src/trade.py jsons/wig_20.json pca QSVC true &
python3 src/trade.py jsons/S\&P.json qpca LinearRegression true &
python3 src/trade.py jsons/wig_20.json qpca LinearRegression true &
python3 src/trade.py jsons/S\&P.json qpca SVR true &
python3 src/trade.py jsons/wig_20.json qpca SVR true &
python3 src/trade.py jsons/S\&P.json qpca QSVR true &
python3 src/trade.py jsons/wig_20.json qpca QSVR true &
python3 src/trade.py jsons/S\&P.json qpca SVC true &
python3 src/trade.py jsons/wig_20.json qpca SVC true &
python3 src/trade.py jsons/S\&P.json qpca QSVC true &
python3 src/trade.py jsons/wig_20.json qpca QSVC true &
wait

#python3 src/predict.py jsons/S\&P.json pca LinearRegression true
#python3 src/predict.py jsons/S\&P.json pca SVC true
#python3 src/predict.py jsons/S\&P.json pca SVR true
#python3 src/predict.py jsons/wig_20.json pca LinearRegression true
#python3 src/predict.py jsons/wig_20.json pca SVC true
#python3 src/predict.py jsons/wig_20.json pca SVR true
#
#
#python3 src/predict.py jsons/S\&P.json pca QSVR true &
#python3 src/predict.py jsons/S\&P.json pca QSVC true &
#python3 src/predict.py jsons/wig_20.json pca QSVC true &
#wait
#python3 src/train.py jsons/wig_20.json pca QSVR true
#python3 src/predict.py jsons/S\&P.json pca QSVR &
#python3 src/predict.py jsons/S\&P.json qpca LinearRegression &
#python3 src/predict.py jsons/S\&P.json qpca SVC &
#python3 src/predict.py jsons/S\&P.json qpca SVR &
#python3 src/predict.py jsons/wig_20.json qpca LinearRegression &
#python3 src/predict.py jsons/wig_20.json qpca SVC &
#python3 src/predict.py jsons/wig_20.json qpca SVR &


#wait

#python3 src/train.py jsons/S\&P.json qpca QSVC true
#echo "Start of train S&P PCA"
#python3 src/train.py jsons/S\&P.json qpca SVC true
#echo "Start of train S&P QPCA"
#python3 src/train.py jsons/S\&P.json qpca
#echo "Start of train S&P SVC"
#python3 src/train.py jsons/S\&P.json svc
#echo "Start of train S&P QSVC"
#python3 src/train.py jsons/S\&P.json qsvc
#echo "Start of predictions S&P PCA"
#python3 src/predict.py jsons/S\&P.json pca
#echo "Start of predictions S&P QPCA"
#python3 src/predict.py jsons/S\&P.json qpca
#echo "Start of train wig_20 PCA"
#python3 src/train.py jsons/wig_20.json pca
#echo "Start of train wig_20 SVR"
#python3 src/train.py jsons/wig_20.json svr_reg
#echo "Start of train wig_20 QPCA"
#python3 src/train.py jsons/wig_20.json qpca
#echo "Start of train wig_20 SVC"
#python3 src/train.py jsons/wig_20.json svc
#echo "Start of predictions wig_20 PCA"
#python3 src/predict.py jsons/wig_20.json pca
#echo "Start of predictions wig_20 QPCA"
#python3 src/predict.py jsons/wig_20.json qpca
#echo "Start of predictions wig_20 SVC"
#python3 src/predict.py jsons/wig_20.json svc
#echo "Start of predictions wig_20 QSVC"
#python3 src/predict.py jsons/wig_20.json qsvc
#echo "Start of predictions wig_20 QPCA"
#python3 src/predict.py jsons/wig_20.json qpca_reg
#echo "Start of predictions wig_20 QSVR"
#python3 src/predict.py jsons/wig_20.json qsvr_reg
#src/venv/bin/python3 src/main.py jsons/wig_20.json
#echo "Start of tests NVDA"
#echo "Start of train NVDA PCA"
#python3 src/train.py jsons/NVDA.json pca_reg
#echo "Start of train NVDA SVR"
#python3 src/train.py jsons/NVDA.json svr_reg
#echo "Start of train NVDA QPCA"
#python3 src/train.py jsons/NVDA.json qpca_reg
#echo "Start of train NVDA QSVR"
#python3 src/train.py jsons/NVDA.json qsvr_reg
#echo "Start of predictions NVDA PCA"
#python3 src/predict.py jsons/NVDA.json pca_reg
#echo "Start of predictions NVDA SVR"
#python3 src/predict.py jsons/NVDA.json svr_reg
#echo "Start of predictions NVDA QPCA"
#python3 src/predict.py jsons/NVDA.json qpca_reg
#echo "Start of predictions NVDA QSVR"
#python3 src/predict.py jsons/NVDA.json qsvr_reg
#src/venv/bin/python3 src/main.py jsons/NVDA.json
#echo "Start of tests CDR"
#src/venv/bin/python3 src/main.py jsons/CDR.json
#echo "Start of tests NFLX"
#src/venv/bin/python3 src/main.py jsons/NFLX.json
#echo "Start of tests PKO"
#src/venv/bin/python3 src/main.py jsons/PKO.json
#echo "Start of tests TSLA"
#src/venv/bin/python3 src/main.py jsons/TSLA.json
#echo "Start of tests MSFT"
#src/venv/bin/python3 src/main.py jsons/MSFT.json
#echo "Start of tests META"
#src/venv/bin/python3 src/main.py jsons/META.json
#echo "Start of tests AAPL"
#src/venv/bin/python3 src/main.py jsons/AAPL.json
#echo "Start of tests ADBE"
#src/venv/bin/python3 src/main.py jsons/ADBE.json
#echo "Start of tests AMZN"
#src/venv/bin/python3 src/main.py jsons/AMZN.json
#echo "Start of tests BAC"
#src/venv/bin/python3 src/main.py jsons/BAC.json
#echo "Start of tests BRK-B"
#src/venv/bin/python3 src/main.py jsons/BRK-B.json
#echo "Start of tests COST"
#src/venv/bin/python3 src/main.py jsons/COST.json
#echo "Start of tests CVX"
#src/venv/bin/python3 src/main.py jsons/CVX.json
#echo "Start of tests HD"
#src/venv/bin/python3 src/main.py jsons/HD.json
echo "End of tests"
deactivate
