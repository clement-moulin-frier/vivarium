from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AgentConfig(_message.Message):
    __slots__ = ["base_length", "neighbor_radius", "proxs_cos_min", "proxs_dist_max", "speed_mul", "theta_mul", "wheel_diameter"]
    BASE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_RADIUS_FIELD_NUMBER: _ClassVar[int]
    PROXS_COS_MIN_FIELD_NUMBER: _ClassVar[int]
    PROXS_DIST_MAX_FIELD_NUMBER: _ClassVar[int]
    SPEED_MUL_FIELD_NUMBER: _ClassVar[int]
    THETA_MUL_FIELD_NUMBER: _ClassVar[int]
    WHEEL_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    base_length: float
    neighbor_radius: float
    proxs_cos_min: float
    proxs_dist_max: float
    speed_mul: float
    theta_mul: float
    wheel_diameter: float
    def __init__(self, wheel_diameter: _Optional[float] = ..., base_length: _Optional[float] = ..., speed_mul: _Optional[float] = ..., theta_mul: _Optional[float] = ..., neighbor_radius: _Optional[float] = ..., proxs_dist_max: _Optional[float] = ..., proxs_cos_min: _Optional[float] = ...) -> None: ...

class AgentConfigSerialized(_message.Message):
    __slots__ = ["serialized"]
    SERIALIZED_FIELD_NUMBER: _ClassVar[int]
    serialized: str
    def __init__(self, serialized: _Optional[str] = ...) -> None: ...

class IsStartedState(_message.Message):
    __slots__ = ["is_started"]
    IS_STARTED_FIELD_NUMBER: _ClassVar[int]
    is_started: bool
    def __init__(self, is_started: bool = ...) -> None: ...

class NDArray(_message.Message):
    __slots__ = ["ndarray"]
    NDARRAY_FIELD_NUMBER: _ClassVar[int]
    ndarray: bytes
    def __init__(self, ndarray: _Optional[bytes] = ...) -> None: ...

class Name(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class PopulationConfig(_message.Message):
    __slots__ = ["n_agents"]
    N_AGENTS_FIELD_NUMBER: _ClassVar[int]
    n_agents: int
    def __init__(self, n_agents: _Optional[int] = ...) -> None: ...

class PopulationConfigSerialized(_message.Message):
    __slots__ = ["serialized"]
    SERIALIZED_FIELD_NUMBER: _ClassVar[int]
    serialized: str
    def __init__(self, serialized: _Optional[str] = ...) -> None: ...

class Position(_message.Message):
    __slots__ = ["x", "y"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: _containers.RepeatedScalarFieldContainer[float]
    y: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, x: _Optional[_Iterable[float]] = ..., y: _Optional[_Iterable[float]] = ...) -> None: ...

class SerializedDict(_message.Message):
    __slots__ = ["entity_behaviors", "has_entity_behaviors", "serialized_dict"]
    ENTITY_BEHAVIORS_FIELD_NUMBER: _ClassVar[int]
    HAS_ENTITY_BEHAVIORS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_DICT_FIELD_NUMBER: _ClassVar[int]
    entity_behaviors: NDArray
    has_entity_behaviors: bool
    serialized_dict: str
    def __init__(self, serialized_dict: _Optional[str] = ..., has_entity_behaviors: bool = ..., entity_behaviors: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class SimulationConfig(_message.Message):
    __slots__ = ["box_size", "dt", "entity_behaviors", "freq", "map_dim", "n_agents", "num_lax_loops", "num_steps_lax", "to_jit", "use_fori_loop"]
    BOX_SIZE_FIELD_NUMBER: _ClassVar[int]
    DT_FIELD_NUMBER: _ClassVar[int]
    ENTITY_BEHAVIORS_FIELD_NUMBER: _ClassVar[int]
    FREQ_FIELD_NUMBER: _ClassVar[int]
    MAP_DIM_FIELD_NUMBER: _ClassVar[int]
    NUM_LAX_LOOPS_FIELD_NUMBER: _ClassVar[int]
    NUM_STEPS_LAX_FIELD_NUMBER: _ClassVar[int]
    N_AGENTS_FIELD_NUMBER: _ClassVar[int]
    TO_JIT_FIELD_NUMBER: _ClassVar[int]
    USE_FORI_LOOP_FIELD_NUMBER: _ClassVar[int]
    box_size: float
    dt: float
    entity_behaviors: NDArray
    freq: float
    map_dim: int
    n_agents: int
    num_lax_loops: int
    num_steps_lax: int
    to_jit: bool
    use_fori_loop: bool
    def __init__(self, box_size: _Optional[float] = ..., map_dim: _Optional[int] = ..., num_steps_lax: _Optional[int] = ..., num_lax_loops: _Optional[int] = ..., dt: _Optional[float] = ..., freq: _Optional[float] = ..., to_jit: bool = ..., n_agents: _Optional[int] = ..., entity_behaviors: _Optional[_Union[NDArray, _Mapping]] = ..., use_fori_loop: bool = ...) -> None: ...

class SimulationConfigSenderName(_message.Message):
    __slots__ = ["config", "name"]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    config: SimulationConfig
    name: Name
    def __init__(self, config: _Optional[_Union[SimulationConfig, _Mapping]] = ..., name: _Optional[_Union[Name, _Mapping]] = ...) -> None: ...

class SimulationConfigSerialized(_message.Message):
    __slots__ = ["serialized"]
    SERIALIZED_FIELD_NUMBER: _ClassVar[int]
    serialized: str
    def __init__(self, serialized: _Optional[str] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ["positions", "thetas"]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    THETAS_FIELD_NUMBER: _ClassVar[int]
    positions: Position
    thetas: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, positions: _Optional[_Union[Position, _Mapping]] = ..., thetas: _Optional[_Iterable[float]] = ...) -> None: ...

class StateArrays(_message.Message):
    __slots__ = ["entity_type", "motors", "positions", "proxs", "thetas"]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    MOTORS_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    PROXS_FIELD_NUMBER: _ClassVar[int]
    THETAS_FIELD_NUMBER: _ClassVar[int]
    entity_type: int
    motors: NDArray
    positions: NDArray
    proxs: NDArray
    thetas: NDArray
    def __init__(self, positions: _Optional[_Union[NDArray, _Mapping]] = ..., thetas: _Optional[_Union[NDArray, _Mapping]] = ..., proxs: _Optional[_Union[NDArray, _Mapping]] = ..., motors: _Optional[_Union[NDArray, _Mapping]] = ..., entity_type: _Optional[int] = ...) -> None: ...

class Time(_message.Message):
    __slots__ = ["time"]
    TIME_FIELD_NUMBER: _ClassVar[int]
    time: int
    def __init__(self, time: _Optional[int] = ...) -> None: ...
