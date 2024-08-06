import os
import sys
import asyncio

# Add submodules to the system path
sys.path.append(r'submodules')

# Import the specific function or module from bleakheart
from bleakheart.examples import ecg_rr_acc_recorder as recorder # type: ignore

async def main():
    # Call the main function from ecg_queue, assuming it's an async function
    await recorder.main()

# Use asyncio.run() to run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())
