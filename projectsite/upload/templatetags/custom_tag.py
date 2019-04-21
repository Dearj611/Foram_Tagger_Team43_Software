from django import template

register = template.Library()   

@register.filter
def index(List, i):
    return List[int(i)]


@register.filter
def boundary(arr, current):
    end = max(arr)
    start = 1
    if current-4 > 1:
        start = current-4
    if current+4 < end:
        end = current+4
    return [i for i in range(start,end+1)]