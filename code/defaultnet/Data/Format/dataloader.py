from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate
import random

class DataLoader(torchDataLoader):
    """ Enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Example:
        >>> class CustomSet(ln.data.Dataset):
        ...     def __len__(self):
        ...         return 4
        ...     @ln.data.Dataset.resize_getitem
        ...     def __getitem__(self, index):
        ...         # Should return (image, anno) but here we return (input_dim,)
        ...         return (self.input_dim,)
        >>> dl = ln.data.DataLoader(
        ...     CustomSet((200,200)),
        ...     batch_size = 2,
        ...     collate_fn = ln.data.list_collate   # We want the data to be grouped as a list
        ... )
        >>> dl.dataset.input_dim    # Default input_dim
        (200, 200)
        >>> for d in dl:
        ...     d
        [[(200, 200), (200, 200)]]
        [[(200, 200), (200, 200)]]
        >>> dl.change_input_dim(320, random_range=(1, 1))
        >>> for d in dl:
        ...     d
        [[(320, 320), (320, 320)]]
        [[(320, 320), (320, 320)]]
        >>> dl.change_input_dim((480, 320), random_range=(1, 1))
        >>> for d in dl:
        ...     d
        [[(480, 320), (480, 320)]]
        [[(480, 320), (480, 320)]]
    """
    def __init__(self, *args, resize_range=(10, 19), **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        sampler = None
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']
        elif len(args) > 3:
            shuffle = args[2]
            if 'sampler' in kwargs:
                sampler = kwargs['sampler']
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']
        else:
            if 'shuffle' in kwargs:
                shuffle = kwargs['shuffle']
            if 'sampler' in kwargs:
                sampler = kwargs['sampler']
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last, input_dimension=self.dataset.input_dim)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

        self.__initialized = True

    def change_input_dim(self, multiple=32, random_range=(10, 20), finish=False):
        """ This function will compute a new size and update it on the next mini_batch.

        Args:
            multiple (int or tuple, optional): value (or values) to multiply the randomly generated range by; Default **32**
            random_range (tuple, optional): This (min, max) tuple sets the range for the randomisation; Default **(10, 19)**

        Note:
            The new size is generated as follows: |br|
            First we compute a random integer inside ``[random_range]``.
            We then multiply that number with the ``multiple`` argument, which gives our final new input size. |br|
            If ``multiple`` is an integer we generate a square size. If you give a tuple of **(width, height)**,
            the size is computed as :math:`rng * multiple[0], rng * multiple[1]`.
        """
        if finish:
            size = random_range[1]
        else:
            size = random.randint(*random_range)

        if isinstance(multiple, int):
            size = (size * multiple, size * multiple)
        else:
            size = (size * multiple[0], size * multiple[1])

        #size = (416, 416)
        self.batch_sampler.new_input_dim = size


class BatchSampler(torchBatchSampler):
    """ This batch sampler will generate mini-batches of (dim, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`, but it will prepend a dimension,
    whilst ensuring it stays the same across one mini-batch.
    """
    def __init__(self, *args, input_dimension=None, **kwargs):
        super(BatchSampler, self).__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None

    def __iter__(self):
        self.__set_input_dim()
        for batch in super(BatchSampler, self).__iter__():
            yield [(self.input_dim, idx) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """ This function randomly changes the the input dimension of the dataset. """
        if self.new_input_dim is not None:
            log.info(f'Resizing network {self.new_input_dim[:2]}')
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None


def list_collate(batch):
    """ Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):
            items[i] = list(items[i])
        else:
            items[i] = default_collate(items[i])

    return items
