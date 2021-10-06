from rectpack.maxrects import MaxRectsBssf

import operator
import itertools
import collections

import decimal


class BinFactory(object):

    def __init__(self, width, height, count, pack_algo, *args, **kwargs):
        self._width = width
        self._height = height
        self._count = count
        
        self._pack_algo = pack_algo
        self._algo_kwargs = kwargs
        self._algo_args = args
        self._ref_bin = None # Reference bin used to calculate fitness
        
        self._bid = kwargs.get("bid", None)

    def _create_bin(self):
        return self._pack_algo(self._width, self._height, *self._algo_args, **self._algo_kwargs)

    def is_empty(self):
        return self._count<1

    def fitness(self, width, height):
        if not self._ref_bin:
            self._ref_bin = self._create_bin()

        return self._ref_bin.fitness(width, height)

    def fits_inside(self, width, height):
        # Determine if rectangle widthxheight will fit into empty bin
        if not self._ref_bin:
            self._ref_bin = self._create_bin()

        return self._ref_bin._fits_surface(width, height)

    def new_bin(self):
        if self._count > 0:
            self._count -= 1
            return self._create_bin()
        else:
            return None

    def __eq__(self, other):
        return self._width*self._height == other._width*other._height

    def __lt__(self, other):
        return self._width*self._height < other._width*other._height

    def __str__(self):
        return "Bin: {} {} {}".format(self._width, self._height, self._count)
        
        
class PackerOnline(object):
    """
    Rectangles are packed as soon are they are added
    """

    def __init__(self, pack_algo=MaxRectsBssf, rotation=False):
        """
        Arguments:
            pack_algo (PackingAlgorithm): What packing algo to use
            rotation (bool): Enable/Disable rectangle rotation
        """
        self._rotation = rotation
        self._pack_algo = pack_algo
        self.reset()

    def __iter__(self):
        return itertools.chain(self._closed_bins, self._open_bins)

    def __len__(self):
        return len(self._closed_bins)+len(self._open_bins)
    
    def __getitem__(self, key):
        """
        Return bin in selected position. (excluding empty bins)
        """
        if not isinstance(key, int):
            raise TypeError("Indices must be integers")

        size = len(self)  # avoid recalulations

        if key < 0:
            key += size

        if not 0 <= key < size:
            raise IndexError("Index out of range")
        
        if key < len(self._closed_bins):
            return self._closed_bins[key]
        else:
            return self._open_bins[key-len(self._closed_bins)]

    def _new_open_bin(self, width=None, height=None, rid=None):
        """
        Extract the next empty bin and append it to open bins
        Returns:
            PackingAlgorithm: Initialized empty packing bin.
            None: No bin big enough for the rectangle was found
        """
        factories_to_delete = set() #
        new_bin = None

        for key, binfac in self._empty_bins.items():

            # Only return the new bin if the rect fits.
            # (If width or height is None, caller doesn't know the size.)
            if not binfac.fits_inside(width, height):
                continue
           
            # Create bin and add to open_bins
            new_bin = binfac.new_bin()
            if new_bin is None:
                continue
            self._open_bins.append(new_bin)

            # If the factory was depleted mark for deletion
            if binfac.is_empty():
                factories_to_delete.add(key)
       
            break

        # Delete marked factories
        for f in factories_to_delete:
            del self._empty_bins[f]

        return new_bin 

    def add_bin(self, width, height, count=1, **kwargs):
        # accept the same parameters as PackingAlgorithm objects
        kwargs['rot'] = self._rotation
        bin_factory = BinFactory(width, height, count, self._pack_algo, **kwargs)
        self._empty_bins[next(self._bin_count)] = bin_factory

    def rect_list(self):
        rectangles = []
        bin_count = 0

        for abin in self:
            for rect in abin:
                rectangles.append((bin_count, rect.x, rect.y, rect.width, rect.height, rect.rid))
            bin_count += 1

        return rectangles

    def bin_list(self):
        """
        Return a list of the dimmensions of the bins in use, that is closed
        or open containing at least one rectangle
        """
        return [(b.width, b.height) for b in self]

    def validate_packing(self):
        for b in self:
            b.validate_packing()

    def reset(self): 
        # Bins fully packed and closed.
        self._closed_bins = collections.deque()

        # Bins ready to pack rectangles
        self._open_bins = collections.deque()

        # User provided bins not in current use
        self._empty_bins = collections.OrderedDict() # O(1) deletion of arbitrary elem
        self._bin_count = itertools.count()


class ShakerBFDProMax(PackerOnline):
    """
    Rectangles aren't packed untils pack() is called
    """
    first_item = operator.itemgetter(0)

    def __init__(self, pack_algo=MaxRectsBssf, sort_key=lambda r: r[0]*r[1],rotation=False):
        """
        """
        super().__init__(pack_algo=pack_algo, rotation=rotation)
        
        self._sort_algo = lambda rectlist: sorted(rectlist, reverse=True, key=sort_key)

        # User provided bins and Rectangles
        self._avail_bins = collections.deque()
        self._avail_rect = collections.deque()

        # Aux vars used during packing
        self._sorted_rect = []

    def add_bin(self, width, height, count=1, **kwargs):
        self._avail_bins.append((width, height, count, kwargs))

    def add_rect(self, width, height, rid=None):
        self._avail_rect.append((width, height, rid))

    def _is_everything_ready(self):
        return self._avail_rect and self._avail_bins
    
    def add_rect_into_bin(self, width, height, rid=None):
 
        # Try packing into open bins
        fit = ((b.fitness(width, height),  b) for b in self._open_bins)
        #print(list(fit))
        fit = (b for b in fit if b[0] is not None)
        try:
            _, best_bin = min(fit, key=self.first_item)
            best_bin.add_rect(width, height, rid)
            return True
        except ValueError:
            return False    


    def pack(self):

        self.reset()

        if not self._is_everything_ready():
            print('Alarm')
            # maybe we should throw an error here?
            return

        # Add available bins to packer
        for b in self._avail_bins:
            width, height, count, extra_kwargs = b
            super().add_bin(width, height, count, **extra_kwargs)

        # If enabled sort rectangles
        self._sorted_rect = self._sort_algo(self._avail_rect)

        # Start packing
        
        # open new bins here!!!!!!!!!!!!!!!!!!!!!!!
        state = 0
        while len(self._sorted_rect) > 0:
            r = self._sorted_rect.pop(0)
            append_success = self.add_rect_into_bin(*r)
            if append_success:
                if state == 0:
                    state = 1
            else:
                if state == 0:
                    new_bin = self._new_open_bin(*r)
                    new_bin.add_rect(*r)
                elif state == 1:
                    state = 2 
                    self._sorted_rect = self._sorted_rect[::-1]
                elif state == 2:
                    self._sorted_rect = self._sorted_rect[::-1]
                    state = 0