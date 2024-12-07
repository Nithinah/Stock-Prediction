Stock Price Prediction Using Flask and LSTM Model
This project presents a web-based stock price prediction system developed using Flask, a lightweight Python web framework, and a Long Short-Term Memory (LSTM) neural network. The system enables users to predict the next day’s closing price of a stock by entering its ticker symbol.

The application fetches historical stock data over the past five years using the Yahoo Finance API and preprocesses it using MinMaxScaler for feature scaling. A deep learning model, specifically an LSTM network, is trained on the historical data to capture temporal dependencies and predict the future stock price. The prediction results, along with a visualization of the historical and predicted prices, are displayed on a user-friendly web interface.

Key features of the system include:

Real-Time Data Retrieval: Dynamic fetching of stock data via Yahoo Finance API.
Deep Learning Prediction: Utilization of LSTM for time-series forecasting.
Visualization: Graphical representation of historical and predicted prices using Matplotlib.
Error Handling: Graceful handling of invalid stock tickers or empty datasets to ensure a seamless user experience.
This system offers a foundational platform for stock price prediction, demonstrating the integration of machine learning models within a web application. It serves as a valuable tool for individuals and organizations seeking to analyze and forecast stock market trends efficiently.

![image](https://github.com/user-attachments/assets/6f63cea7-71ca-4caf-b6e3-580f914aa038)

![image](https://github.com/user-attachments/assets/db4928cf-17c1-4bc8-a8bb-499748838248)

