import collections

class MetricsHistoryLogger():
    """
    This is a logger that logs the metrics history, since PyTorch Lightning does not directly support this feature
    It logs the metrics history in `self.history`, as long as the metric name does not end with '_auto_max' or '_auto_min'
    Refer to trainer.log_max_and_min_metrics() for automatically adding the max and min metrics
    """
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list) # copy not necessary here  
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "metrics_log_collector"

    @property
    def version(self):
        return "1.0"

    @property
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            self.history[metric_name].append(metric_value)
        return
