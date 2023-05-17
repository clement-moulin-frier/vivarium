from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AgentConfig(_message.Message):
    __slots__ = ["base_length", "behavior", "entity_type", "proxs_cos_min", "proxs_dist_max", "speed_mul", "theta_mul", "wheel_diameter"]
    BASE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROXS_COS_MIN_FIELD_NUMBER: _ClassVar[int]
    PROXS_DIST_MAX_FIELD_NUMBER: _ClassVar[int]
    SPEED_MUL_FIELD_NUMBER: _ClassVar[int]
    THETA_MUL_FIELD_NUMBER: _ClassVar[int]
    WHEEL_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    base_length: float
    behavior: str
    entity_type: int
    proxs_cos_min: float
    proxs_dist_max: float
    speed_mul: float
    theta_mul: float
    wheel_diameter: float
    def __init__(self, wheel_diameter: _Optional[float] = ..., base_length: _Optional[float] = ..., speed_mul: _Optional[float] = ..., theta_mul: _Optional[float] = ..., proxs_dist_max: _Optional[float] = ..., proxs_cos_min: _Optional[float] = ..., behavior: _Optional[str] = ..., entity_type: _Optional[int] = ...) -> None: ...

class AgentConfigIdxSenderName(_message.Message):
    __slots__ = ["config", "idx", "name"]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    IDX_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    config: AgentConfig
    idx: AgentIdx
    name: Name
    def __init__(self, config: _Optional[_Union[AgentConfig, _Mapping]] = ..., name: _Optional[_Union[Name, _Mapping]] = ..., idx: _Optional[_Union[AgentIdx, _Mapping]] = ...) -> None: ...

class AgentConfigSerialized(_message.Message):
    __slots__ = ["serialized"]
    SERIALIZED_FIELD_NUMBER: _ClassVar[int]
    serialized: str
    def __init__(self, serialized: _Optional[str] = ...) -> None: ...

class AgentConfigs(_message.Message):
    __slots__ = ["agent_configs"]
    AGENT_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    agent_configs: _containers.RepeatedCompositeFieldContainer[AgentConfig]
    def __init__(self, agent_configs: _Optional[_Iterable[_Union[AgentConfig, _Mapping]]] = ...) -> None: ...

class AgentIdx(_message.Message):
    __slots__ = ["idx"]
    IDX_FIELD_NUMBER: _ClassVar[int]
    idx: int
    def __init__(self, idx: _Optional[int] = ...) -> None: ...

class Behaviors(_message.Message):
    __slots__ = ["agent_slice", "behavior"]
    AGENT_SLICE_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
    agent_slice: Slice
    behavior: str
    def __init__(self, agent_slice: _Optional[_Union[Slice, _Mapping]] = ..., behavior: _Optional[str] = ...) -> None: ...

class IsStartedState(_message.Message):
    __slots__ = ["is_started"]
    IS_STARTED_FIELD_NUMBER: _ClassVar[int]
    is_started: bool
    def __init__(self, is_started: bool = ...) -> None: ...

class Motors(_message.Message):
    __slots__ = ["agent_slice", "motors"]
    AGENT_SLICE_FIELD_NUMBER: _ClassVar[int]
    MOTORS_FIELD_NUMBER: _ClassVar[int]
    agent_slice: Slice
    motors: NDArray
    def __init__(self, agent_slice: _Optional[_Union[Slice, _Mapping]] = ..., motors: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class NDArray(_message.Message):
    __slots__ = ["ndarray"]
    NDARRAY_FIELD_NUMBER: _ClassVar[int]
    ndarray: bytes
    def __init__(self, ndarray: _Optional[bytes] = ...) -> None: ...

class NVEState(_message.Message):
    __slots__ = ["base_length", "behavior", "entity_type", "force", "mass", "momentum", "motor", "position", "prox", "proxs_cos_min", "proxs_dist_max", "speed_mul", "theta_mul", "wheel_diameter"]
    BASE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    MASS_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    MOTOR_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    PROXS_COS_MIN_FIELD_NUMBER: _ClassVar[int]
    PROXS_DIST_MAX_FIELD_NUMBER: _ClassVar[int]
    PROX_FIELD_NUMBER: _ClassVar[int]
    SPEED_MUL_FIELD_NUMBER: _ClassVar[int]
    THETA_MUL_FIELD_NUMBER: _ClassVar[int]
    WHEEL_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    base_length: NDArray
    behavior: NDArray
    entity_type: NDArray
    force: RigidBody
    mass: RigidBody
    momentum: RigidBody
    motor: NDArray
    position: RigidBody
    prox: NDArray
    proxs_cos_min: NDArray
    proxs_dist_max: NDArray
    speed_mul: NDArray
    theta_mul: NDArray
    wheel_diameter: NDArray
    def __init__(self, position: _Optional[_Union[RigidBody, _Mapping]] = ..., momentum: _Optional[_Union[RigidBody, _Mapping]] = ..., force: _Optional[_Union[RigidBody, _Mapping]] = ..., mass: _Optional[_Union[RigidBody, _Mapping]] = ..., prox: _Optional[_Union[NDArray, _Mapping]] = ..., motor: _Optional[_Union[NDArray, _Mapping]] = ..., behavior: _Optional[_Union[NDArray, _Mapping]] = ..., wheel_diameter: _Optional[_Union[NDArray, _Mapping]] = ..., base_length: _Optional[_Union[NDArray, _Mapping]] = ..., speed_mul: _Optional[_Union[NDArray, _Mapping]] = ..., theta_mul: _Optional[_Union[NDArray, _Mapping]] = ..., proxs_dist_max: _Optional[_Union[NDArray, _Mapping]] = ..., proxs_cos_min: _Optional[_Union[NDArray, _Mapping]] = ..., entity_type: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

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

class RigidBody(_message.Message):
    __slots__ = ["center", "orientation"]
    CENTER_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    center: NDArray
    orientation: NDArray
    def __init__(self, center: _Optional[_Union[NDArray, _Mapping]] = ..., orientation: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

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
    __slots__ = ["box_size", "dt", "freq", "map_dim", "neighbor_radius", "num_lax_loops", "num_steps_lax", "to_jit", "use_fori_loop"]
    BOX_SIZE_FIELD_NUMBER: _ClassVar[int]
    DT_FIELD_NUMBER: _ClassVar[int]
    FREQ_FIELD_NUMBER: _ClassVar[int]
    MAP_DIM_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_RADIUS_FIELD_NUMBER: _ClassVar[int]
    NUM_LAX_LOOPS_FIELD_NUMBER: _ClassVar[int]
    NUM_STEPS_LAX_FIELD_NUMBER: _ClassVar[int]
    TO_JIT_FIELD_NUMBER: _ClassVar[int]
    USE_FORI_LOOP_FIELD_NUMBER: _ClassVar[int]
    box_size: float
    dt: float
    freq: float
    map_dim: int
    neighbor_radius: float
    num_lax_loops: int
    num_steps_lax: int
    to_jit: bool
    use_fori_loop: bool
    def __init__(self, box_size: _Optional[float] = ..., map_dim: _Optional[int] = ..., num_steps_lax: _Optional[int] = ..., num_lax_loops: _Optional[int] = ..., dt: _Optional[float] = ..., freq: _Optional[float] = ..., neighbor_radius: _Optional[float] = ..., to_jit: bool = ..., use_fori_loop: bool = ...) -> None: ...

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

class SimulationConfigWithOneOf(_message.Message):
    __slots__ = ["box_size_value", "dt_value", "freq_value", "has_box_size", "has_dt", "has_freq", "has_map_dim", "has_neighbor_radius", "has_num_lax_loops", "has_num_steps_lax", "has_to_jit", "has_use_fori_loop", "map_dim_value", "neighbor_radius_value", "num_lax_loops_value", "num_steps_lax_value", "to_jit_value", "use_fori_loop_value"]
    BOX_SIZE_VALUE_FIELD_NUMBER: _ClassVar[int]
    DT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FREQ_VALUE_FIELD_NUMBER: _ClassVar[int]
    HAS_BOX_SIZE_FIELD_NUMBER: _ClassVar[int]
    HAS_DT_FIELD_NUMBER: _ClassVar[int]
    HAS_FREQ_FIELD_NUMBER: _ClassVar[int]
    HAS_MAP_DIM_FIELD_NUMBER: _ClassVar[int]
    HAS_NEIGHBOR_RADIUS_FIELD_NUMBER: _ClassVar[int]
    HAS_NUM_LAX_LOOPS_FIELD_NUMBER: _ClassVar[int]
    HAS_NUM_STEPS_LAX_FIELD_NUMBER: _ClassVar[int]
    HAS_TO_JIT_FIELD_NUMBER: _ClassVar[int]
    HAS_USE_FORI_LOOP_FIELD_NUMBER: _ClassVar[int]
    MAP_DIM_VALUE_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_RADIUS_VALUE_FIELD_NUMBER: _ClassVar[int]
    NUM_LAX_LOOPS_VALUE_FIELD_NUMBER: _ClassVar[int]
    NUM_STEPS_LAX_VALUE_FIELD_NUMBER: _ClassVar[int]
    TO_JIT_VALUE_FIELD_NUMBER: _ClassVar[int]
    USE_FORI_LOOP_VALUE_FIELD_NUMBER: _ClassVar[int]
    box_size_value: float
    dt_value: float
    freq_value: float
    has_box_size: bool
    has_dt: bool
    has_freq: bool
    has_map_dim: bool
    has_neighbor_radius: bool
    has_num_lax_loops: bool
    has_num_steps_lax: bool
    has_to_jit: bool
    has_use_fori_loop: bool
    map_dim_value: int
    neighbor_radius_value: float
    num_lax_loops_value: int
    num_steps_lax_value: int
    to_jit_value: bool
    use_fori_loop_value: bool
    def __init__(self, has_box_size: bool = ..., box_size_value: _Optional[float] = ..., has_map_dim: bool = ..., map_dim_value: _Optional[int] = ..., has_num_steps_lax: bool = ..., num_steps_lax_value: _Optional[int] = ..., has_num_lax_loops: bool = ..., num_lax_loops_value: _Optional[int] = ..., has_dt: bool = ..., dt_value: _Optional[float] = ..., has_freq: bool = ..., freq_value: _Optional[float] = ..., has_neighbor_radius: bool = ..., neighbor_radius_value: _Optional[float] = ..., has_to_jit: bool = ..., to_jit_value: bool = ..., has_use_fori_loop: bool = ..., use_fori_loop_value: bool = ...) -> None: ...

class Slice(_message.Message):
    __slots__ = ["start", "step", "stop"]
    START_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    start: int
    step: int
    stop: int
    def __init__(self, start: _Optional[int] = ..., stop: _Optional[int] = ..., step: _Optional[int] = ...) -> None: ...

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
