import pandas as pd
import numpy as np

class DataUtils:

	def __init__(self, data_path_final= 'data/customers-final-2.csv'):
		self.features = {'time_delta':-3,'Tickets':-2,'csat':-1}
		print("loading csv")
		customers_final = pd.read_csv(data_path_final)

		print("removing columns")
		fields_to_drop = ['Unnamed: 0','MRR', 'Seats','days']
		data = customers_final.drop(fields_to_drop, axis=1)

		quant_features = ['time_delta','Tickets','csat']

		self.scaled_features = {}

		print("scaling features")
		for each in quant_features:
		    mean, std = np.float32(data[each].mean()), np.float32(data[each].std())
		    self.scaled_features[each] = [mean, std]
		
		print(self.scaled_features)

	def scale_data(self,input_data):
		m_time_delta, s_time_delta = self.scaled_features['time_delta']
		m_tickets, s_tickets = self.scaled_features['Tickets']
		m_csat, s_csat = self.scaled_features['csat']

		input_data[-3] = np.float32(np.float32((input_data[-3] - m_time_delta)) / s_time_delta)
		input_data[-2] = np.float32(np.float32((input_data[-2] - m_tickets)) / s_tickets)
		input_data[-1] = np.float32(np.float32((input_data[-1] - m_csat)) / s_csat)

		print("Scaling data")

		return input_data

	def downscale_data(self,input_data):
		m_time_delta, s_time_delta = self.scaled_features['time_delta']
		m_tickets, s_tickets = self.scaled_features['Tickets']
		m_csat, s_csat = self.scaled_features['csat']

		input_data[-3] = np.float32(np.float32(input_data[-3]*s_time_delta) + m_time_delta)
		input_data[-2] = np.float32(np.float32(input_data[-2]*s_tickets) + m_tickets)
		input_data[-1] = np.float32(np.float32(input_data[-1]*s_csat) + m_csat)

		print("downscaling data")

		return input_data