# evaluate
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
end_time = time.time()

print("time: ",end_time-start_time)
print(f"LOSS: {loss}\nR2:  {r2}")
model.save(path+f"model_save/r2_{r2:.4f}.h5")

# 위에서 y까지 minmax해버렸기에 inverse_transform 해주기
predicted_degC = minmax_for_y.inverse_transform(np.array(y_predict).reshape(-1,1))
y_true = minmax_for_y.inverse_transform(np.array(y_test).reshape(-1,1))
print(x_test.shape,y_predict.shape,predicted_degC.shape)

# 실제로 잘 나온건지 원 데이터와 비교하기 위한 csv파일 생성
submit = pd.DataFrame(np.array([y_true,predicted_degC]).reshape(-1,2),columns=['true','predict'])
submit.to_csv(path+f"submit_r2_{r2}.csv",index=False)

# LOSS: 1.1545700544957072e-05
# R2:  0.9994092879419377