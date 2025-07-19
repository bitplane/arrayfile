# 🪵🪵🪵

> What rolls down stairs, alone or in pairs, and over your neighbor's dog?
> What's great for a snack, And fits on your back?
> It's Log Log Log!

I needed line wrapping in my tty with fast access, so why not make a log that
supports egregious sizes and low seek times?

## ⏩

```bash
# makes log.log.log from /var/log
make log
```

## TODO

- [x] Rename to logloglog
- [x] Cache module
  - [x] Cleanup old caches
  - [x] Allocate new cache
  - [x] Better cache names like file_name[{md5sum(inode + ctime)[:8]}].ext
  - [ ] Store path in data file
- [ ] Features
  - [ ] Pane (tail with width and height)
  - [ ] Scrolly (pane + vertical scrollbar)
  - [ ] Scrollx (no wrap, char wrap, word wrap)
  - [ ] Search
- [ ] Line formatters (strip ansi, plain, jsonpp...)
- [ ] Async/non-blocking design
  - [ ] Make log processing async to avoid blocking
  - [ ] Support streaming updates
- [ ] Interface design and view mixins
  - [ ] Proper view mixin architecture
  - [ ] Pipeable view types (search, filter, transform)
- [ ] Multiple display backends
  - [ ] WebSocket/FastAPI for web UIs
  - [ ] Pipe commands for shell integration
  - [ ] Rich for direct terminal display
  - [ ] Textual for interactive TUIs
  - [ ] Curses/urwid fallbacks
- [ ] Python logging integration
  - [ ] Direct logger handlers
  - [ ] Log level filtering and formatting
- [ ] Cache management
  - [ ] Periodic cleanup of cache directories
  - [ ] Check file existence by inode lookup
  - [ ] Handle file rotation and moves
