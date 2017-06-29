# Simple script which deletes tf$contrib$distributions as for some reason 
# pkgdown breaks the custom tf distribution aliases
sed -i "s/tf\$contrib\$distributions\\$<span/<span/g" ./articles/*.html
