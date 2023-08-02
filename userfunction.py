def evaluate_model(xtrain,ytrain,xtest,ytest,model):
    from sklearn.metrics import mean_absolute_error, r2_score
    # Predict train and test values
    ypred_tr = model.predict(xtrain)
    ypred_ts = model.predict(xtest)
    
    # Evaluate for training
    tr_mae = mean_absolute_error(ytrain,ypred_tr)
    tr_r2 = r2_score(ytrain, ypred_tr)
    
    # Evaluate for testing
    ts_mae = mean_absolute_error(ytest,ypred_ts)
    ts_r2 = r2_score(ytest, ypred_ts)
    
    # Print the results
    print('Training Results:')
    print(f'MAE  : {tr_mae:.2f}')
    print(f'R2   : {tr_r2:.4f}')
    
    print('\n=======================\n')
    print('Testing Results:')
    print(f'MAE  : {ts_mae:.2f}')
    print(f'R2   : {ts_r2:.4f}')