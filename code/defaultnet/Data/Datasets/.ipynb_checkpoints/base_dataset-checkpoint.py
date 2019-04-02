from functools import wraps
import torch
from torch.utils.data.dataset import Dataset as torchDataset

class Dataset(torchDataset):
    """ This class is a subclass of the base :class:`torch.utils.data.Dataset`,
    that enables on the fly resizing of the ``input_dim`` with :class:`lightnet.data.DataLoader`.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
    """
    def __init__(self, input_dimension):
        super().__init__()
        self.__input_dim = input_dimension[:2]

    @property
    def input_dim(self):
        """ Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, '_input_dim'):
            return self._input_dim
        return self.__input_dim

    @staticmethod
    def resize_getitem(getitem_fn):
        """ Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the on the fly resizing  of the ``input_dim`` with our :class:`~lightnet.data.DataLoader` class.

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.resize_getitem
            ...     def __getitem__(self, index):
            ...         # Should return (image, anno) but here we return input_dim
            ...         return self.input_dim
            >>> data = CustomSet((200,200))
            >>> data[0]
            (200, 200)
            >>> data[(480,320), 0]
            (480, 320)
        """
        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self._input_dim = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            if hasattr(self, '_input_dim'):
                del self._input_dim

            return ret_val

        return wrapper
