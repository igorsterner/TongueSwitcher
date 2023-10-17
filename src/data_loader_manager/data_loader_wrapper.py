import logging

from easydict import EasyDict

logger = logging.getLogger(__name__)

class DataLoaderWrapper:
    """
    Data loader wrapper, general class definitions
    """

    def __init__(self, config):
        self.config = config

    def build_dataset(self):
        """
        This function loads data and features required for building the dataset
        """

        self.data = EasyDict()
        dataset_modules = self.config.data_loader.dataset_modules.module_list
        print(dataset_modules)
        print(self.config.data_loader.dataset_modules.module_dict)
        for dataset_module in dataset_modules:
            module_config = (
                self.config.data_loader.dataset_modules.module_dict[
                    dataset_module
                ]
            )
            loading_func = getattr(self, dataset_module)
            loading_func(module_config)
            print("data columns: {}".format(self.data.keys()))
