# gpu
Rust library to apply oprations on a buffer using the gpu
```rust
use gpu::{calc, execute_gpu};

fn main() {
    let spirv = calc!(b0 += b1 * b2 + b3 + 4);
    let r = futures::executor::block_on(execute_gpu(
        vec![vec![0, 0, 0], vec![5, 4, 0], vec![1, 2, 0], vec![2, 5, 4]],
        spirv.as_binary(),
    ));
    assert_eq!(r, vec!(11, 17, 8));
}
```

***execute_gpu***

Takes a vector of buffers, the first buffer is used as the output buffer. 

Takes the compiled spirv as well and run the compiled operations.

The user must ensure that the spirv and the vector have the same number of buffers.

***calc***

Macro to compile spirv from an equation
All buffers need to have the name b with a digit like b1
b0 is the output buffer
Examples:
```rust
  calc!(b0 = 4)
  calc!(b0 += 4)
  calc!(b0 += b1 * b2 + 5)
```
