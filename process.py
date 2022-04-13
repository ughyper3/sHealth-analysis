from pandas import DataFrame


class ProcessDateset:

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
        As there are few missing values compared to the number of values,
        we can fill the missing values with the mean of the column.
        :param data: input dataset
        :return: output dataset
        """
        filled_data_set = data.fillna(data.mean())
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
    def process_data(data: DataFrame) -> DataFrame:
        """
        process the data by applying the methods created above.
        :param data: input dataset
        :return: output processed dataset
        """
        # todo: rename step attributes, not sure if its a good practice or not
        step_1 = ProcessDateset.drop_sleep_time_column(data)
        step_2 = ProcessDateset.process_day_column(step_1)
        step_3 = ProcessDateset.process_week_column(step_2)
        step_4 = ProcessDateset.fill_missing_values(step_3)
        step_5 = ProcessDateset.round_columns(step_4)
        step_6 = ProcessDateset.rename_columns(step_5)
        return step_6

