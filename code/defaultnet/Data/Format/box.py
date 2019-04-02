__all__=['Box']

class Box:
    """ This is a generic bounding box representation.
    This class provides some base functionality to both annotations and detections.

    Attributes:
        class_label (string): class string label; Default **''**
        object_id (int): Object identifier for reid purposes; Default **0**
        x_top_left (Number): X pixel coordinate of the top left corner of the bounding box; Default **0.0**
        y_top_left (Number): Y pixel coordinate of the top left corner of the bounding box; Default **0.0**
        width (Number): Width of the bounding box in pixels; Default **0.0**
        height (Number): Height of the bounding box in pixels; Default **0.0**
    """
    def __init__(self):
        self.class_label = ''   # class string label
        self.object_id = 0      # object identifier
        self.x_top_left = 0.0   # x pixel coordinate top left of the box
        self.y_top_left = 0.0   # y pixel coordinate top left of the box
        self.width = 0.0        # width of the box in pixels
        self.height = 0.0       # height of the box in pixels

    @classmethod
    def create(cls, obj=None):
        """ Create a bounding box from a string or other detection object.

        Args:
            obj (Box or string, optional): Bounding box object to copy attributes from or string to deserialize
        """
        instance = cls()

        if obj is None:
            return instance

        if isinstance(obj, str):
            instance.deserialize(obj)
        elif isinstance(obj, Box):
            instance.class_label = obj.class_label
            instance.object_id = obj.object_id
            instance.x_top_left = obj.x_top_left
            instance.y_top_left = obj.y_top_left
            instance.width = obj.width
            instance.height = obj.height
        else:
            raise TypeError(f'Object is not of type Box or not a string [obj.__class__.__name__]')

        return instance

    def __eq__(self, other):
        # TODO: refactor -> use almost equal for floats
        return self.__dict__ == other.__dict__

    def serialize(self):
        """ abstract serializer, implement in derived classes. """
        raise NotImplementedError

    def deserialize(self, string):
        """ abstract parser, implement in derived classes. """
        raise NotImplementedError

