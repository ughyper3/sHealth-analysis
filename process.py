from pandas import DataFrame, date_range, to_datetime, DatetimeIndex


class ProcessDataset:

    @staticmethod
    def set_datetime_column(raw_data: DataFrame) -> DataFrame:
        """
        Convert the date column into a datetime object column, be careful of the format of the datetime object
        as the pandas parser was confused by the dd/mm/yyyy format, precise dayfirst attribute is True
        :param raw_data:
        :return: raw data
        """
        raw_data["Date"] = to_datetime(raw_data["Date"], dayfirst=True)
        return raw_data

    @staticmethod
    def drop_sleep_time_column(raw_data: DataFrame) -> DataFrame:
        """
        Remove sleep time column because there is more than 70% of missing values
        :param raw_data: input dataset
        :return: output dataset without the sleep time class
        """
        data_set = raw_data.drop(['sleep time'], axis=1)
        return data_set

    @staticmethod
    def process_day_column(data: DataFrame) -> DataFrame:
        """
        Get days of the week from the date column in order to be sure
         that the days of the week given corresponds to the date
        :param data: input dataset
        :return: output dataset
        """
        data.Day = data.Date.dt.day_name()
        return data

    @staticmethod
    def process_week_column(data: DataFrame) -> DataFrame:
        """
        Get days of the week from the date column in order to be sure
         that the days of the week given corresponds to the date
        :param data: input dataset
        :return: output dataset
        """
        data['week'] = data['Date'].dt.strftime('%Y-%U')
        return data

    @staticmethod
    def fill_missing_values(data: DataFrame) -> DataFrame:
        """
        instead of fill the dataset with mean, we choose to delete the missing rows
        :param data: input dataset
        :return: output dataset
        """
        filled_data_set = data.dropna()
        return filled_data_set

    @staticmethod
    def round_columns(data: DataFrame) -> DataFrame:
        """
        round numerical columns
        :param data: input dataset
        :return: output rounded dataset
        """
        return data.round(decimals=0)

    @staticmethod
    def rename_columns(raw_data: DataFrame) -> DataFrame:
        """
        Renaming columns by following good naming practices
        :param raw_data: input dataset
        :return: output dataset
        """
        data_set = raw_data.rename(
            columns={
                'Day': 'day',
                'Date': 'date',
                'Weigth': 'weight',
                'Steps': 'steps',
                'Walk (min)': 'walk_duration',
                'Sport (min)': 'sport_duration',
                'Spent kcal': 'spent_energy',
                'GPS country': 'country_number'
            }
        )
        return data_set

    @staticmethod
    def sanity_check_duplicate_date(raw_data: DataFrame) -> bool:
        """
        This method checks if there is no duplication and no missing values in the date column
        by comparing the list of date with a list of date range between the min and the max date of our dataset
        and return a boolean in response
        :param raw_data:
        :return: boolean
        """
        list_of_removed_dates = ['2015-07-27', '2017-01-16', '2017-01-05', '2016-11-13', '2015-10-10', '2015-07-04', '2017-01-13', '2017-01-17', '2017-01-06', '2017-01-07', '2017-01-08', '2017-01-11', '2015-09-12', '2017-01-09', '2017-01-14', '2015-10-25', '2017-01-03', '2017-01-04', '2017-01-15', '2017-01-01', '2017-01-12'].sort()
        date = date_range(raw_data['date'].min(), raw_data['date'].max()).to_series()
        raw_data = list(raw_data['date'])
        date_list = []
        raw_data_list = []
        for date in date:
            date_list.append(date.strftime('%Y-%m-%d'))
        for date in raw_data:
            raw_data_list.append(date.strftime('%Y-%m-%d'))
        difference = list(set(date_list) - set(raw_data_list)).sort()

        if difference == list_of_removed_dates:
            response = f'Dates columns are ready to be used'
            print(response)
            return True
        else:
            response = f'Dates columns are not ready to be used, difference : {difference}'
            print(response)
            return False

    @staticmethod
    def create_average_speed_column(raw_data: DataFrame) -> DataFrame:
        """
        calculation on the average walk speed based on the steps, the sport and the daily walk duration
        we supposed that the average foot walk length is 0.64m for a human
        :param raw_data:
        :return: data set with average speed column
        """
        steps_by_minute = raw_data['steps'] / (raw_data['sport_duration'] + raw_data['walk_duration'])
        steps_by_hour = steps_by_minute * 60
        meter_by_hour = steps_by_hour * 0.64
        kilometer_by_hour = meter_by_hour / 1000
        raw_data['average_speed'] = kilometer_by_hour
        return raw_data

    @staticmethod
    def set_date_as_dataset_index(raw_data: DataFrame) -> DataFrame:
        """
        As we have to work with a time series analysis, we should set the date as the dataset index
        :param raw_data: input dataset
        :return:
        """
        raw_data.index = raw_data.date
        raw_data.index = DatetimeIndex(raw_data.index).to_period('D')
        return raw_data

    @staticmethod
    def process_data(data: DataFrame) -> DataFrame:
        """
        process the data by applying the methods created above.
        :param data: input dataset
        :return: output processed dataset
        """
        try:
            step_0 = ProcessDataset.set_datetime_column(data)
            step_1 = ProcessDataset.drop_sleep_time_column(step_0)
            step_2 = ProcessDataset.process_day_column(step_1)
            step_3 = ProcessDataset.process_week_column(step_2)
            step_4 = ProcessDataset.fill_missing_values(step_3)
            step_5 = ProcessDataset.round_columns(step_4)
            step_6 = ProcessDataset.rename_columns(step_5)
            step_7 = ProcessDataset.create_average_speed_column(step_6)
            step_8 = ProcessDataset.set_date_as_dataset_index(step_7)
            ProcessDataset.sanity_check_duplicate_date(step_8)

        except Exception as e:
            print(f'exception {e} during the data preprocessing')
            return data
        else:
            print(f'Successful data preprocessing')
            return step_8


