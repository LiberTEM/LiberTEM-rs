import libertem_dectris

if __name__ == "__main__":
    frames = libertem_dectris.FrameChunkedIterator()
    frames.start()

    frame_stack = frames.get_next_stack(max_size=32)
    serialized = frame_stack.serialize()
    loaded = libertem_dectris.FrameStack.deserialize(serialized)

    print(serialized)
    print(len(loaded))
    print(loaded[0])
    print(loaded[0].get_pixel_type())
    print(loaded[0].get_encoding())
    print(loaded[0].get_shape())
