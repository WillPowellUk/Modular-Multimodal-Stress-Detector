import os
import sys
import asyncio

# Add submodules to the system path
sys.path.append(r'submodules')

# Import the specific function or module from bleakheart
from bleakheart.examples import ecg_queue # type: ignore

async def main():
    # Call the main function from ecg_queue, assuming it's an async function
    await ecg_queue.main()

# Use asyncio.run() to run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())
