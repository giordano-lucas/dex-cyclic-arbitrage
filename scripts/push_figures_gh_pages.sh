git pull
git checkout gh-pages
git pull
git checkout main -- figures
git add figures
git commit -m "Update 'figures' directory from 'main' branch"
git checkout main