import logging

logger = logging.getLogger(__name__)


from tongueswitcher.saving_processors import SavingProcessor
from utils.dirs import *


class BaseExecutor(SavingProcessor):
    additional_plugins = []
    
    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.data_loader = data_loader
        
        logger.info(f'Initializing {self.__class__.__name__}...')
    
    def setup(self, stage):
        """
        set loggers as class attributes for easy access
        """
        for trainer_logger in self.trainer.loggers:
            if type(trainer_logger) == TensorBoardLogger:
                self.tb_logger = trainer_logger
            elif type(trainer_logger) == WandbLogger:
                self.wandb_logger = trainer_logger
                self.wandb_logger.watch(self.model, log_freq=500, log_graph=False)
            elif type(trainer_logger) == MetricsHistoryLogger:
                self.metrics_history_logger = trainer_logger
            else:
                logger.warning(f'Unsupported logger type: {type(trainer_logger)}')