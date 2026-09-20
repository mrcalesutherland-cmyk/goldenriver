[app]
title = Golden River Oracle
package.name = goldenriver
package.domain = org.goldenriver
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,pkl
version = 1.0.0
requirements = python3,kivy,numpy,pandas,scipy,matplotlib,requests,urllib3,certifi

orientation = portrait
fullscreen = 0
android.permissions = INTERNET
android. accept\_sdk\_license = True
# Android target settings
android.api = 34
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
