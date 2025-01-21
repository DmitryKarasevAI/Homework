from aiogram.fsm.state import State, StatesGroup


class SetUpForm(StatesGroup):
    weight = State()
    height = State()
    age = State()
    sex = State()
    active_time = State()
    city = State()
