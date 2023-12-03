import sys
import inspect
import time

func_call_counter = 0
func_tracing_execution_times = []
function_tracing = True


def isPrivate(func_name):
    if func_name[0:2] == '__':
        return True
    else:
        return False


def FUNCTION_TRACE_BEGIN():
    if not function_tracing:
        return

    global func_call_counter
    global func_tracing_execution_times

    tab = func_call_counter * ' '

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
            print('%s%d BEGIN PRIVATE METHOD %s.%s' % (tab, func_call_counter, class_name, func_name))
        else:
            print('%s%d BEGIN PUBLIC  METHOD %s.%s' % (tab, func_call_counter, class_name, func_name))
    else:
        print('%s%d BEGIN FUNCTION %s' % (tab, func_call_counter, func_name))

    func_call_counter += 1
    func_tracing_execution_times.append(time.time())


def FUNCTION_TRACE_END():
    if not function_tracing:
        return

    global func_call_counter
    global func_tracing_execution_times

    time_elapsed = time.time() - func_tracing_execution_times[-1]

    func_call_counter -= 1
    tab = func_call_counter * ' '

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
            print('%s%d END   PRIVATE METHOD %s.%s (%.2f s)' % (tab, func_call_counter, class_name, func_name, time_elapsed))
        else:
            print('%s%d END   PUBLIC  METHOD %s.%s (%.2f s)' % (tab, func_call_counter, class_name, func_name, time_elapsed))
    else:
        print('%s%d END   FUNCTION %s (%.2f s)' % (tab, func_call_counter, func_name, time_elapsed))

    func_tracing_execution_times.pop()
