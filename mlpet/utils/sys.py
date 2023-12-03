
import inspect
import time

# globals
_func_call_counter = 0
_func_tracing_execution_times = []
_function_tracing = True


def isPrivate(func_name) -> bool:

    if func_name[0:2] == '__':
        return True
    else:
        return False


def FUNCTION_TRACE_BEGIN() -> None:

    if not _function_tracing:
        return

    global _func_call_counter
    global _func_tracing_execution_times

    tab = _func_call_counter * ' '

    stack = inspect.stack()[1][0]
    func_name = stack.f_code.co_name

    if 'self' in stack.f_locals:
        class_name = stack.f_locals['self'].__class__.__name__
        if isPrivate(func_name):
            func_name = func_name[2:]
            private = True
        else:
            private = False
        if private:
            print('%s%d BEGIN PRIVATE METHOD %s.%s' % (tab, _func_call_counter, class_name, func_name))
        else:
            print('%s%d BEGIN PUBLIC  METHOD %s.%s' % (tab, _func_call_counter, class_name, func_name))
    else:
        print('%s%d BEGIN FUNCTION %s' % (tab, _func_call_counter, func_name))

    _func_call_counter += 1
    _func_tracing_execution_times.append(time.time())


def FUNCTION_TRACE_END() -> None:

    if not _function_tracing:
        return

    global _func_call_counter
    global _func_tracing_execution_times

    time_elapsed = time.time() - _func_tracing_execution_times[-1]

    _func_call_counter -= 1
    tab = _func_call_counter * ' '

    stack = inspect.stack()[1][0]
    func_name = stack.f_code.co_name
    if 'self' in stack.f_locals:
        class_name = stack.f_locals['self'].__class__.__name__
        if isPrivate(func_name):
            func_name = func_name[2:]
            private = True
        else:
            private = False
        if private:
            print('%s%d END   PRIVATE METHOD %s.%s (%.2f s)' % (tab, _func_call_counter, class_name, func_name, time_elapsed))
        else:
            print('%s%d END   PUBLIC  METHOD %s.%s (%.2f s)' % (tab, _func_call_counter, class_name, func_name, time_elapsed))
    else:
        print('%s%d END   FUNCTION %s (%.2f s)' % (tab, _func_call_counter, func_name, time_elapsed))

    _func_tracing_execution_times.pop()
