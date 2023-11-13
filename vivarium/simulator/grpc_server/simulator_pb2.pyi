from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Name(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class Time(_message.Message):
    __slots__ = ["time"]
    TIME_FIELD_NUMBER: _ClassVar[int]
    time: int
    def __init__(self, time: _Optional[int] = ...) -> None: ...

class Slice(_message.Message):
    __slots__ = ["start", "stop", "step"]
    START_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    start: int
    stop: int
    step: int
    def __init__(self, start: _Optional[int] = ..., stop: _Optional[int] = ..., step: _Optional[int] = ...) -> None: ...

class SerializedDict(_message.Message):
    __slots__ = ["serialized"]
    SERIALIZED_FIELD_NUMBER: _ClassVar[int]
    serialized: str
    def __init__(self, serialized: _Optional[str] = ...) -> None: ...

class SimulationConfig(_message.Message):
    __slots__ = ["box_size", "map_dim", "n_agents", "num_steps_lax", "num_lax_loops", "dt", "freq", "neighbor_radius", "to_jit", "use_fori_loop"]
    BOX_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAP_DIM_FIELD_NUMBER: _ClassVar[int]
    N_AGENTS_FIELD_NUMBER: _ClassVar[int]
    NUM_STEPS_LAX_FIELD_NUMBER: _ClassVar[int]
    NUM_LAX_LOOPS_FIELD_NUMBER: _ClassVar[int]
    DT_FIELD_NUMBER: _ClassVar[int]
    FREQ_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_RADIUS_FIELD_NUMBER: _ClassVar[int]
    TO_JIT_FIELD_NUMBER: _ClassVar[int]
    USE_FORI_LOOP_FIELD_NUMBER: _ClassVar[int]
    box_size: float
    map_dim: int
    n_agents: int
    num_steps_lax: int
    num_lax_loops: int
    dt: float
    freq: float
    neighbor_radius: float
    to_jit: bool
    use_fori_loop: bool
    def __init__(self, box_size: _Optional[float] = ..., map_dim: _Optional[int] = ..., n_agents: _Optional[int] = ..., num_steps_lax: _Optional[int] = ..., num_lax_loops: _Optional[int] = ..., dt: _Optional[float] = ..., freq: _Optional[float] = ..., neighbor_radius: _Optional[float] = ..., to_jit: bool = ..., use_fori_loop: bool = ...) -> None: ...

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

class AgentIdx(_message.Message):
    __slots__ = ["idx"]
    IDX_FIELD_NUMBER: _ClassVar[int]
    idx: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, idx: _Optional[_Iterable[int]] = ...) -> None: ...

class AgentConfig(_message.Message):
    __slots__ = ["wheel_diameter", "base_length", "speed_mul", "theta_mul", "proxs_dist_max", "proxs_cos_min", "behavior", "color", "entity_type"]
    WHEEL_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    BASE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    SPEED_MUL_FIELD_NUMBER: _ClassVar[int]
    THETA_MUL_FIELD_NUMBER: _ClassVar[int]
    PROXS_DIST_MAX_FIELD_NUMBER: _ClassVar[int]
    PROXS_COS_MIN_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    wheel_diameter: float
    base_length: float
    speed_mul: float
    theta_mul: float
    proxs_dist_max: float
    proxs_cos_min: float
    behavior: str
    color: str
    entity_type: int
    def __init__(self, wheel_diameter: _Optional[float] = ..., base_length: _Optional[float] = ..., speed_mul: _Optional[float] = ..., theta_mul: _Optional[float] = ..., proxs_dist_max: _Optional[float] = ..., proxs_cos_min: _Optional[float] = ..., behavior: _Optional[str] = ..., color: _Optional[str] = ..., entity_type: _Optional[int] = ...) -> None: ...

class AgentConfigIdxSenderName(_message.Message):
    __slots__ = ["config", "name", "idx"]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IDX_FIELD_NUMBER: _ClassVar[int]
    config: AgentConfig
    name: Name
    idx: AgentIdx
    def __init__(self, config: _Optional[_Union[AgentConfig, _Mapping]] = ..., name: _Optional[_Union[Name, _Mapping]] = ..., idx: _Optional[_Union[AgentIdx, _Mapping]] = ...) -> None: ...

class SerializedDictSenderName(_message.Message):
    __slots__ = ["dict", "name"]
    DICT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    dict: SerializedDict
    name: Name
    def __init__(self, dict: _Optional[_Union[SerializedDict, _Mapping]] = ..., name: _Optional[_Union[Name, _Mapping]] = ...) -> None: ...

class SerializedDictIdxSenderName(_message.Message):
    __slots__ = ["dict", "name", "idx"]
    DICT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IDX_FIELD_NUMBER: _ClassVar[int]
    dict: SerializedDict
    name: Name
    idx: AgentIdx
    def __init__(self, dict: _Optional[_Union[SerializedDict, _Mapping]] = ..., name: _Optional[_Union[Name, _Mapping]] = ..., idx: _Optional[_Union[AgentIdx, _Mapping]] = ...) -> None: ...

class AgentConfigs(_message.Message):
    __slots__ = ["agent_configs"]
    AGENT_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    agent_configs: _containers.RepeatedCompositeFieldContainer[AgentConfig]
    def __init__(self, agent_configs: _Optional[_Iterable[_Union[AgentConfig, _Mapping]]] = ...) -> None: ...

class AllConfigSerialized(_message.Message):
    __slots__ = ["simulation_config", "agent_configs", "object_configs"]
    SIMULATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    AGENT_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    OBJECT_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    simulation_config: str
    agent_configs: str
    object_configs: str
    def __init__(self, simulation_config: _Optional[str] = ..., agent_configs: _Optional[str] = ..., object_configs: _Optional[str] = ...) -> None: ...

class AgentConfigSerialized(_message.Message):
    __slots__ = ["serialized"]
    SERIALIZED_FIELD_NUMBER: _ClassVar[int]
    serialized: str
    def __init__(self, serialized: _Optional[str] = ...) -> None: ...

class AgentConfigsSerialized(_message.Message):
    __slots__ = ["serialized"]
    SERIALIZED_FIELD_NUMBER: _ClassVar[int]
    serialized: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, serialized: _Optional[_Iterable[str]] = ...) -> None: ...

class NDArray(_message.Message):
    __slots__ = ["ndarray"]
    NDARRAY_FIELD_NUMBER: _ClassVar[int]
    ndarray: bytes
    def __init__(self, ndarray: _Optional[bytes] = ...) -> None: ...

class RigidBody(_message.Message):
    __slots__ = ["center", "orientation"]
    CENTER_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    center: NDArray
    orientation: NDArray
    def __init__(self, center: _Optional[_Union[NDArray, _Mapping]] = ..., orientation: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class NVEState(_message.Message):
    __slots__ = ["position", "momentum", "force", "mass", "diameter", "entity_type", "entity_idx", "friction"]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    MASS_FIELD_NUMBER: _ClassVar[int]
    DIAMETER_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_IDX_FIELD_NUMBER: _ClassVar[int]
    FRICTION_FIELD_NUMBER: _ClassVar[int]
    position: RigidBody
    momentum: RigidBody
    force: RigidBody
    mass: RigidBody
    diameter: NDArray
    entity_type: NDArray
    entity_idx: NDArray
    friction: NDArray
    def __init__(self, position: _Optional[_Union[RigidBody, _Mapping]] = ..., momentum: _Optional[_Union[RigidBody, _Mapping]] = ..., force: _Optional[_Union[RigidBody, _Mapping]] = ..., mass: _Optional[_Union[RigidBody, _Mapping]] = ..., diameter: _Optional[_Union[NDArray, _Mapping]] = ..., entity_type: _Optional[_Union[NDArray, _Mapping]] = ..., entity_idx: _Optional[_Union[NDArray, _Mapping]] = ..., friction: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class AgentState(_message.Message):
    __slots__ = ["nve_idx", "prox", "motor", "behavior", "wheel_diameter", "speed_mul", "theta_mul", "proxs_dist_max", "proxs_cos_min", "color"]
    NVE_IDX_FIELD_NUMBER: _ClassVar[int]
    PROX_FIELD_NUMBER: _ClassVar[int]
    MOTOR_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
    WHEEL_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    SPEED_MUL_FIELD_NUMBER: _ClassVar[int]
    THETA_MUL_FIELD_NUMBER: _ClassVar[int]
    PROXS_DIST_MAX_FIELD_NUMBER: _ClassVar[int]
    PROXS_COS_MIN_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    nve_idx: NDArray
    prox: NDArray
    motor: NDArray
    behavior: NDArray
    wheel_diameter: NDArray
    speed_mul: NDArray
    theta_mul: NDArray
    proxs_dist_max: NDArray
    proxs_cos_min: NDArray
    color: NDArray
    def __init__(self, nve_idx: _Optional[_Union[NDArray, _Mapping]] = ..., prox: _Optional[_Union[NDArray, _Mapping]] = ..., motor: _Optional[_Union[NDArray, _Mapping]] = ..., behavior: _Optional[_Union[NDArray, _Mapping]] = ..., wheel_diameter: _Optional[_Union[NDArray, _Mapping]] = ..., speed_mul: _Optional[_Union[NDArray, _Mapping]] = ..., theta_mul: _Optional[_Union[NDArray, _Mapping]] = ..., proxs_dist_max: _Optional[_Union[NDArray, _Mapping]] = ..., proxs_cos_min: _Optional[_Union[NDArray, _Mapping]] = ..., color: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class ObjectState(_message.Message):
    __slots__ = ["nve_idx", "custom_field", "color"]
    NVE_IDX_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_FIELD_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    nve_idx: NDArray
    custom_field: NDArray
    color: NDArray
    def __init__(self, nve_idx: _Optional[_Union[NDArray, _Mapping]] = ..., custom_field: _Optional[_Union[NDArray, _Mapping]] = ..., color: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ["nve_state", "agent_state", "object_state"]
    NVE_STATE_FIELD_NUMBER: _ClassVar[int]
    AGENT_STATE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_STATE_FIELD_NUMBER: _ClassVar[int]
    nve_state: NVEState
    agent_state: AgentState
    object_state: ObjectState
    def __init__(self, nve_state: _Optional[_Union[NVEState, _Mapping]] = ..., agent_state: _Optional[_Union[AgentState, _Mapping]] = ..., object_state: _Optional[_Union[ObjectState, _Mapping]] = ...) -> None: ...

class MotorInfo(_message.Message):
    __slots__ = ["agent_idx", "motor_idx", "value"]
    AGENT_IDX_FIELD_NUMBER: _ClassVar[int]
    MOTOR_IDX_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    agent_idx: AgentIdx
    motor_idx: int
    value: float
    def __init__(self, agent_idx: _Optional[_Union[AgentIdx, _Mapping]] = ..., motor_idx: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...

class Motor(_message.Message):
    __slots__ = ["agent_idx", "motor"]
    AGENT_IDX_FIELD_NUMBER: _ClassVar[int]
    MOTOR_FIELD_NUMBER: _ClassVar[int]
    agent_idx: int
    motor: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, agent_idx: _Optional[int] = ..., motor: _Optional[_Iterable[float]] = ...) -> None: ...

class Prox(_message.Message):
    __slots__ = ["agent_idx", "prox"]
    AGENT_IDX_FIELD_NUMBER: _ClassVar[int]
    PROX_FIELD_NUMBER: _ClassVar[int]
    agent_idx: int
    prox: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, agent_idx: _Optional[int] = ..., prox: _Optional[_Iterable[float]] = ...) -> None: ...

class Behavior(_message.Message):
    __slots__ = ["agent_idx", "function"]
    AGENT_IDX_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    agent_idx: int
    function: bytes
    def __init__(self, agent_idx: _Optional[int] = ..., function: _Optional[bytes] = ...) -> None: ...

class StateChange(_message.Message):
    __slots__ = ["nve_idx", "col_idx", "nested_field", "value"]
    NVE_IDX_FIELD_NUMBER: _ClassVar[int]
    COL_IDX_FIELD_NUMBER: _ClassVar[int]
    NESTED_FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    nve_idx: _containers.RepeatedScalarFieldContainer[int]
    col_idx: _containers.RepeatedScalarFieldContainer[int]
    nested_field: _containers.RepeatedScalarFieldContainer[str]
    value: NDArray
    def __init__(self, nve_idx: _Optional[_Iterable[int]] = ..., col_idx: _Optional[_Iterable[int]] = ..., nested_field: _Optional[_Iterable[str]] = ..., value: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class AddAgentInput(_message.Message):
    __slots__ = ["n_agents", "serialized_config"]
    N_AGENTS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_CONFIG_FIELD_NUMBER: _ClassVar[int]
    n_agents: int
    serialized_config: str
    def __init__(self, n_agents: _Optional[int] = ..., serialized_config: _Optional[str] = ...) -> None: ...

class IsStartedState(_message.Message):
    __slots__ = ["is_started"]
    IS_STARTED_FIELD_NUMBER: _ClassVar[int]
    is_started: bool
    def __init__(self, is_started: bool = ...) -> None: ...
