# A walkthrough of the C++ in `CanBus.h`

This is a learning-oriented companion to [`include/CanBus.h`](../include/CanBus.h). It assumes you're comfortable with the basics of C++ — classes, methods, pointers, the difference between `.cpp` and `.h` files — but haven't necessarily met the modern-C++ techniques the wrapper leans on. Each section introduces an idea on its own with small standalone examples, then ties it back to the line in `CanBus.h` that uses it.

The big picture: FlexCAN_T4's receive-callback API hands us a single bare-C function pointer with no context attached. Bridging that to a clean object-oriented dispatcher with multiple buses requires a few tricks that are hard to recognize the first time you see them. By the end of this tutorial those tricks should feel routine.

---

## 1. Function pointers and `using` aliases

A C++ function has a *type*, just like an `int` or a `std::string`. The type encodes the return type and parameter list. You can store the address of a function in a variable whose type matches:

```cpp
int add(int a, int b) { return a + b; }

int (*op)(int, int) = &add;   // op is a pointer to a function
int x = op(2, 3);             // calls add(2, 3) → 5
```

The syntax `int (*op)(int, int)` is unfortunately gnarly. The standard escape hatch is a *type alias* — modern C++ uses `using` for this:

```cpp
using BinaryOp = int(*)(int, int);   // BinaryOp is now a type, "pointer to int(int,int)"

BinaryOp op = &add;
int x = op(2, 3);
```

Now read `CanBus.h` line:

```cpp
using Handler = void(*)(const CAN_message_t &msg, void *context);
```

`Handler` is a type alias for "pointer to a function that takes a `const CAN_message_t&` and a `void*`, and returns `void`". Anywhere you'd otherwise write `void(*)(const CAN_message_t&, void*)`, you can now just write `Handler`.

> **Why use `using` instead of the older `typedef`?** They're nearly equivalent, but `using` reads left-to-right (`name = type`), composes better with templates, and is the modern style. For function-pointer types in particular, `using` is dramatically more readable than `typedef`.

---

## 2. The motivation: a callback with no "this" pointer

The FlexCAN_T4 library lets you register a function to be called for every received frame:

```cpp
void onFrame(const CAN_message_t &msg) {
  Serial.println(msg.id, HEX);
}

Can2.onReceive(&onFrame);
```

Look at the type: `void(*)(const CAN_message_t&)`. There's nowhere to pass extra data. If you have two `CanBus` *objects* and you want each one to know "the frame that just arrived was for me," there's no field in that signature where the library can hand you a pointer back to your object. Compare this to a hypothetical API that *did* support context:

```cpp
// Hypothetical, not what FlexCAN_T4 actually does:
Can2.onReceive(&onFrame, /*context=*/ &myCanBusInstance);
```

If FlexCAN looked like that, `onFrame` could just receive `myCanBusInstance` as a parameter and call `myCanBusInstance.dispatch(msg)`. But it doesn't, so we need to invent that mechanism ourselves. That mechanism is the centerpiece of `CanBus.h`.

---

## 3. Templates: a one-paragraph refresher

A *template* is a recipe the compiler uses to generate a concrete piece of code, parameterized by a *type* or a *value*. The classic example:

```cpp
template <typename T>
T max(T a, T b) {
  return (a > b) ? a : b;
}

int  i = max(3, 7);          // compiler stamps out max<int>
float f = max(1.5f, 2.5f);   // compiler stamps out max<float>
```

`template <typename T>` says: "for any type `T`, the following is the recipe." The angle brackets after the function name (`max<int>`) are where you'd specify `T` explicitly, but most of the time the compiler infers it from the argument types.

Templates can also be parameterized on a *value*, not a type. These are called *non-type template parameters*:

```cpp
template <int N>
int multiply_by(int x) { return x * N; }

int a = multiply_by<2>(7);   // compiler stamps out a function that returns x*2
int b = multiply_by<10>(7);  // and a separate function that returns x*10
```

Crucially, `multiply_by<2>` and `multiply_by<10>` are *different functions* with different addresses. The compiler generates one specialized copy per distinct value of `N`. We're going to use this property heavily in the trampoline section.

---

## 4. A function template that wraps an init routine

Sometimes you want a method that works for several closely-related types but you don't want polymorphism overhead. CanBus has this case: it accepts any `FlexCAN_T4<CAN1, …>`, `FlexCAN_T4<CAN2, …>`, or `FlexCAN_T4<CAN3, …>` instance. They're all *different types* (because the CAN peripheral is a template parameter on the FlexCAN class), and the methods we need to call (`begin()`, `setMaxMB()`, `enableFIFO()`, `onReceive()`) live on the *derived* class — not on the abstract base — so a plain `FlexCAN_T4_Base&` parameter wouldn't compile.

Solution: make `begin()` itself a function template:

```cpp
template <typename BusT>
void begin(BusT &bus, uint32_t baud = 1000000) {
  bus_ = &bus;                 // store as base-class pointer for runtime use
  bus.begin();                 // these calls compile because BusT is the
  bus.setBaudRate(baud);       // concrete derived type, not the base.
  bus.setMaxMB(16);
  bus.enableFIFO();
  bus.enableFIFOInterrupt();
  bus.onReceive(/* … trampoline … */);
}
```

The compiler sees the call `bus2.begin(Can2);` and stamps out a copy of `begin()` specialized for `BusT = FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16>`. Inside that copy, `bus.setMaxMB(16)` is a normal call on the concrete derived class, no virtual dispatch.

After init, we store `bus_` as a pointer to the abstract base (`FlexCAN_T4_Base*`). At runtime, the wrapper only ever calls methods that *are* on the base (`write()`, `events()`), so a single `CanBus` type can hold any peripheral. This is the best of both worlds: templated init for compile-time access to derived methods, type-erased base storage for runtime polymorphism.

> **Why is this template defined in the header?** Because templates aren't compiled until they're *instantiated* with concrete arguments. The compiler needs to see the full body of `begin()` at every call site so it can stamp out the right specialization. Putting templates in `.cpp` files almost never works without explicit instantiation. This is the main reason `CanBus` and `DmActuator` are header-only.

---

## 5. Non-type template parameters and the trampoline trick

We come to the heart of the puzzle. We need to register a function pointer with `Can2.onReceive(...)`, and that function — when called — must somehow know which `CanBus` object owns this peripheral.

Here's the move: **generate a different function for each `CanBus` instance**, and have each generated function look up "its" `CanBus` from a static table.

```cpp
inline static CanBus *instances_[MAX_BUSES] = {};   // table of CanBus pointers

template <uint8_t N>
static void trampoline(const CAN_message_t &msg) {
  if (instances_[N]) instances_[N]->dispatch(msg);
}
```

Re-read that with the section-3 framing. `trampoline<0>` is one function, `trampoline<1>` is a *different* function, etc. They have different addresses because the compiler stamps out a separate copy for each `N`. The value of `N` is baked in at compile time — there's no parameter being read, no branch, no table lookup *for `N`*. Each version of the function says "look at slot 0" or "look at slot 1" as a constant.

How do we get a `_MB_ptr` to `trampoline<0>` to hand to FlexCAN? We pre-compute an array of them:

```cpp
inline static const _MB_ptr trampolines_[MAX_BUSES] = {
  &CanBus::trampoline<0>,
  &CanBus::trampoline<1>,
  &CanBus::trampoline<2>,
  &CanBus::trampoline<3>,
};
```

Now `trampolines_[0]` is a `void(*)(const CAN_message_t&)` pointing at the first specialization, `trampolines_[1]` at the second, and so on.

The init flow connects the dots:

```cpp
uint8_t slot = next_slot_++;        // claim a unique slot for this instance
instances_[slot] = this;            // remember "I'm at slot N"
bus.onReceive(trampolines_[slot]);  // tell FlexCAN to call trampoline<N>
```

When a frame arrives:

1. FlexCAN calls `trampoline<N>(msg)`.
2. That trampoline looks up `instances_[N]` — which we set to `this` — and calls `dispatch(msg)` on the right `CanBus`.
3. `dispatch` walks the per-bus handler table.

The `void*`-context problem from section 2 is solved without the library having any concept of context. The "context" is encoded in *which trampoline function* we registered.

> **Why this pattern exists at all.** Lambdas with captures (`auto cb = [this](auto &m){ … };`) feel like the natural answer here, but a lambda with captures is *not* convertible to a plain function pointer — the captured state has to live somewhere, and a bare `void(*)(...)` has nowhere to point at it. So we hand-roll the equivalent: each trampoline "captures" its `CanBus` via a static table indexed by a compile-time constant.

---

## 6. Static class members: declaration vs. definition

When you put `static` in front of a class member, you're saying "this isn't per-object; it's shared by every object of the class." So `next_slot_` and `instances_[]` are *one* set of variables for the whole program, regardless of how many `CanBus` objects exist. That's exactly what we want for the slot allocator and the slot→object table.

Historically (pre-C++17) static class members had a wart: declaring them in the header wasn't enough; you also had to *define* them in exactly one `.cpp` file:

```cpp
// CanBus.h
class CanBus {
  static int next_slot_;   // declaration
};

// CanBus.cpp
int CanBus::next_slot_ = 0;   // definition — must appear in exactly one .cpp file
```

Forget the `.cpp` definition and you get a linker error like *"undefined reference to `CanBus::next_slot_`"*. Have it in two `.cpp` files and you get *"multiple definition of `CanBus::next_slot_`"*. For a header-only library this is a real headache: either you drag in a `.cpp` file (defeating the point) or you play tricks with `inline` functions returning function-static variables (gross).

---

## 7. `inline static` (C++17) — the modern fix

C++17 added a tiny but transformative feature: when you mark a static class member as `inline`, you can both declare *and* define it in the header, and the compiler/linker guarantees there's only one across the whole program — no `.cpp` file needed.

```cpp
class CanBus {
  inline static int next_slot_ = 0;     // declaration AND definition, header-only safe
};
```

This is exactly what `CanBus.h` does:

```cpp
inline static CanBus *instances_[MAX_BUSES] = {};
inline static uint8_t next_slot_ = 0;
inline static const _MB_ptr trampolines_[MAX_BUSES] = { /* … */ };
```

This is why [`platformio.ini`](../platformio.ini) forces `-std=gnu++17`: without it, `inline static` is a syntax error and we'd be back to the awkward header/cpp split. (Teensy's default is gnu++14.)

> **Quick aside on the word `inline`.** It used to mean "please consider inlining this function." That meaning is mostly historical — modern compilers ignore the hint. The *useful* meaning today is the linker-level one: "if you see this definition in multiple translation units, it's fine, just merge them into one." That's the meaning being used here.

---

## 8. The `void *` context pattern — type erasure on the cheap

Once a frame is delivered to `CanBus::dispatch`, we want to call user-supplied handlers. Different handlers belong to different objects (e.g. each `DmActuator` has its own `this`), so we need a way to associate a generic function with object-specific data.

The bare-bones C-style answer:

```cpp
using Handler = void(*)(const CAN_message_t&, void *context);

// caller:
bus.registerHandler(0x12, &DmActuator::onRxStatic, this);

// callback shape:
static void onRxStatic(const CAN_message_t &msg, void *ctx) {
  static_cast<DmActuator*>(ctx)->updateFromMessage(msg);
}
```

The handler is a free function (so it has a stable function-pointer type), and the user's per-object data piggybacks through the `void *context` parameter. Inside the handler, a `static_cast` brings the type back. This is sometimes called *type erasure*: at the dispatch boundary, every handler looks the same; on either side of that boundary, the types are fully concrete.

Compared to `std::function`, this approach has zero heap allocation, zero virtual calls, and total transparency to the optimizer. The cost is that you, the caller, take responsibility for matching the `static_cast` type to whatever you passed in — get it wrong and you have undefined behavior. For embedded code talking to one library, that's a fair trade.

> **`static_cast<T*>(p)`** is the explicit cast you use when you know more about a pointer than the type system does. The compiler trusts you. (`reinterpret_cast` is its trashier cousin, used for bit-level reinterpretation; `static_cast` is for type relationships you know to be valid, like `void*` → `DmActuator*`.)

---

## 9. References vs. pointers as members

A small but worth-noting style detail. Some classes store collaborators as references (`Foo &foo_`), some as pointers (`Foo *foo_`). The choice is mostly about lifetime semantics:

- Use a **reference member** when the collaborator is supplied at construction and is guaranteed to outlive the object. References can't be null and can't be reseated, so the contract is "always valid, always the same target." `DmActuator` uses `CanBus &bus_;` for this reason — the bus must exist when the actuator is constructed.
- Use a **pointer member** when the collaborator is set later, can be nullable, or might be reseated. `CanBus` uses `FlexCAN_T4_Base *bus_ = nullptr;` because `bus_` is set inside `begin()`, not at construction.

Both are zero-overhead at runtime. The choice is documentation and correctness, not performance.

---

## 10. Putting it together: a frame's journey through `CanBus.h`

Here's what actually happens, step by step, when a CAN frame arrives:

1. The CAN transceiver shifts the frame into the FlexCAN peripheral's hardware FIFO. The peripheral raises an interrupt.
2. FlexCAN_T4's *interrupt service routine* runs (this is true ISR context — preempts whatever the CPU was doing). It reads the FIFO and pushes the frame onto an internal software queue. **Our code does not run yet.**
3. Eventually `loop()` runs and calls `bus.poll()`, which calls `Can2.events()` on the underlying peripheral. This is in *thread* context — back to normal program flow.
4. `events()` drains the internal software queue. For each frame, it calls the function pointer that was registered via `Can2.onReceive(...)` at init time.
5. That function pointer is `&CanBus::trampoline<N>` for whatever `N` was assigned to this `CanBus` during `begin()`.
6. `trampoline<N>` is a free function with the `_MB_ptr` signature. Its body is `instances_[N]->dispatch(msg)`. So we land back in the right `CanBus` instance.
7. `dispatch(msg)` walks the per-bus handler table looking for entries whose `can_id` matches `msg.id`. For each match, it calls the handler with `(msg, context)`.
8. For frames arriving on a feedback ID, the handler is `DmActuator::onRxStatic` and the context is the `this` of the `DmActuator` that registered it. The static method `static_cast`s the `void*` back to a `DmActuator*` and updates its cached feedback.

Three properties fall out of this design:

- **Adding a joint is one constructor call** — the joint registers itself in step 7's table; nothing else changes.
- **Adding a bus is one `bus.begin(CanX)` call** — a new slot is allocated, a new trampoline is wired up; nothing in `dispatch` changes.
- **Handlers run in thread context.** They run from `bus.poll()` (i.e. `loop()`), not from the FlexCAN ISR. That's why `bus.send()` and `Serial.print()` from inside a handler are safe — the same calls from inside an `IntervalTimer` callback would race FlexCAN's internal state because *that* really is ISR context.

---

## 11. Things we considered and didn't pick

For completeness — when designing a wrapper like this you'll bump into these alternatives:

- **`std::function<void(const CAN_message_t&)>`** as the handler type. Powerful, supports captures, but allocates on the heap when the captured state is too big to fit in its small-buffer optimization, has hidden virtual calls, and bloats the binary. Not suitable for tight embedded loops.
- **A pure-virtual `IFrameHandler` base class.** Would work and is the typical Java/C# pattern. Adds a vtable per handler, and forces you to subclass to register a handler. The free-function-plus-`void*` pattern is more flexible and lighter.
- **Store the bus number, demux in a single trampoline using `msg.bus`.** `CAN_message_t` does have a `bus` field that FlexCAN populates. A single trampoline could `instances_[msg.bus]->dispatch(msg)`. We didn't go that route because it would force the user to tell `CanBus` which bus number it owns, which is information FlexCAN already knows. The slot-based approach makes `bus.begin(Can2)` self-contained.

---

## 12. Recap

The new-to-you parts of `CanBus.h`, in one paragraph:

- A **`using` alias** is the modern way to give a function-pointer type a readable name.
- **Function templates** let one method accept any of the FlexCAN derived types and call methods that only exist on the derived class.
- A **non-type template parameter** (the `N` in `trampoline<N>`) lets you generate a family of distinct concrete functions, each baking in a different compile-time constant.
- **`inline static`** (C++17) is the clean way to put a single shared static variable in a header.
- The **`void *` context pattern** is how you build a callback table without `std::function` and without inheritance.
- **References for non-null compile-time collaborators, pointers for nullable runtime ones** is a useful default rule.

If you can read those lines without flinching, the rest of `CanBus.h` should feel mechanical. A good follow-up exercise: open [`include/DmActuator.h`](../include/DmActuator.h), find the line where it self-registers its handler, and trace through how that hooks into the dispatch table you now understand.
