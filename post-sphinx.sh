cd docs
mv html/* .
rm -rf html

sed -i -e 's/_static/static/g' *.html
sed -i -e 's/_sources/sources/g' *.html
sed -i -e 's/_images/images/g' *.html
mv _static static
mv _sources sources
mv _images images
rm *.html-e