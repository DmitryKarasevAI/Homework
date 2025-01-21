import asyncio
from aiogram import Bot, Dispatcher
from config import BOT_TOKEN
from handlers.start import include_start_router
from handlers.profile import include_profile_router
from middlewares import LoggingMiddleware

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

dp.message.middleware(LoggingMiddleware())
include_start_router(dp)
include_profile_router(dp)


async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
