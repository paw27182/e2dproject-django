from abc import ABC, ABCMeta, abstractmethod  # Abstract Base Class


class DBAbstractClass(metaclass=ABCMeta):
    def __init__(self, dbname, tablename):
        self.dbname = dbname
        self.tablename = tablename

    @abstractmethod
    def get_form_size(self, selected_form=None):
        return 0

    @abstractmethod
    def write_form_items(self, input_data):
        return "OK"

    @abstractmethod
    def read_form_items(self, command):
        return [], []  # key, items

    @abstractmethod
    def drop_form(self):
        pass

    @abstractmethod
    def delete_item(self, record_to_be_processed):
        pass
