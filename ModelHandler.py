import tensorflow as tf
import datetime as dt


class ModelHandler(object):

    __extension = ".ckpt"
    __path = ""

    def __init__(self, absolute_save_path='', model_name='model', use_timestamp=False):
        self.__path = absolute_save_path + model_name
        if use_timestamp:
            self.__path = self.__path + self.__get_current_datetime()
        self.__path = self.__path + self.__extension

    def save_model(self, saver_object, session):
        print ("Saving the model at location %s" % self.__path)
        saved_path = saver_object.save(session, self.__path)
        return saved_path

    def restore_model(self, saver_object, session, log=True):
        if log:
            print ("Restoring the model at location %s" % self.__path)

        try:
            saver_object.restore(session, self.__path)
            return True
        except:
            print ("Failed to locate the saved model at location %s" % self.__path)
            return False

    def get_saved_path(self):
        return self.__path

    def __get_current_datetime(self):
        return '_' + dt.datetime.now().isoformat()
